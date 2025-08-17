"""
Enhanced arXiv Integration for Nano-GraphRAG

Provides advanced search, paper discovery, metadata analysis, and intelligent
recommendations for academic paper insertion into GraphRAG sessions.
"""

import os
import asyncio
import aiohttp
import tempfile
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from textual import on, log
from textual.app import ComposeResult
from textual.widgets import (
    Static, Input, Button, ListView, ListItem, Checkbox, Select, 
    Label, ProgressBar, TextArea, Collapsible, Tabs, TabPane, Tree, TreeNode
)
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.reactive import reactive
from textual.message import Message

try:
    import feedparser
    import requests
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False


@dataclass
class ArxivPaper:
    """Enhanced paper data structure with comprehensive metadata."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    updated: str
    categories: List[str]
    pdf_url: str
    arxiv_url: str
    comment: str = ""
    journal_ref: str = ""
    doi: str = ""
    
    # Enhanced metadata
    citation_count: int = 0
    download_count: int = 0
    file_size: int = 0
    page_count: int = 0
    reading_time_minutes: int = 0
    quality_score: float = 0.0
    relevance_score: float = 0.0
    
    # Relationship data
    references: List[str] = field(default_factory=list)
    cited_by: List[str] = field(default_factory=list)
    related_papers: List[str] = field(default_factory=list)
    
    # User data
    is_selected: bool = False
    is_downloaded: bool = False
    is_read: bool = False
    user_tags: Set[str] = field(default_factory=set)
    user_notes: str = ""
    added_to_collection: Optional[str] = None


@dataclass
class ArxivSearchQuery:
    """Structured search query with advanced filters."""
    text_query: str = ""
    categories: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sort_by: str = "relevance"  # relevance, recent, citations
    max_results: int = 50
    include_older_versions: bool = False
    
    def to_arxiv_query(self) -> str:
        """Convert to arXiv API query string."""
        query_parts = []
        
        if self.text_query:
            query_parts.append(f"all:{self.text_query}")
        
        if self.categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in self.categories)
            query_parts.append(f"({cat_query})")
        
        if self.authors:
            author_query = " OR ".join(f"au:{author}" for author in self.authors)
            query_parts.append(f"({author_query})")
        
        return " AND ".join(query_parts) if query_parts else "all:*"


class ArxivCache:
    """Intelligent caching system for arXiv data."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".elia_cache" / "arxiv"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.papers_cache: Dict[str, ArxivPaper] = {}
        self.search_cache: Dict[str, List[str]] = {}  # query_hash -> paper_ids
        self.load_cache()
    
    def _hash_query(self, query: ArxivSearchQuery) -> str:
        """Generate hash for search query caching."""
        query_str = f"{query.text_query}_{query.categories}_{query.authors}_{query.date_from}_{query.date_to}_{query.sort_by}_{query.max_results}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def load_cache(self):
        """Load cached data from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Reconstruct papers from cached data
                    for paper_data in data.get('papers', []):
                        paper = ArxivPaper(**paper_data)
                        self.papers_cache[paper.arxiv_id] = paper
                    self.search_cache = data.get('searches', {})
        except Exception as e:
            log.error(f"Failed to load arXiv cache: {e}")
    
    def save_cache(self):
        """Save cached data to disk."""
        try:
            data = {
                'papers': [paper.__dict__ for paper in self.papers_cache.values()],
                'searches': self.search_cache,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Failed to save arXiv cache: {e}")
    
    def get_cached_search(self, query: ArxivSearchQuery) -> Optional[List[ArxivPaper]]:
        """Get cached search results."""
        query_hash = self._hash_query(query)
        if query_hash in self.search_cache:
            paper_ids = self.search_cache[query_hash]
            papers = [self.papers_cache[pid] for pid in paper_ids if pid in self.papers_cache]
            if papers:
                return papers
        return None
    
    def cache_search_results(self, query: ArxivSearchQuery, papers: List[ArxivPaper]):
        """Cache search results."""
        query_hash = self._hash_query(query)
        paper_ids = []
        
        for paper in papers:
            self.papers_cache[paper.arxiv_id] = paper
            paper_ids.append(paper.arxiv_id)
        
        self.search_cache[query_hash] = paper_ids
        self.save_cache()
    
    def get_paper(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Get cached paper by ID."""
        return self.papers_cache.get(arxiv_id)
    
    def update_paper(self, paper: ArxivPaper):
        """Update cached paper data."""
        self.papers_cache[paper.arxiv_id] = paper
        self.save_cache()


class EnhancedArxivAPI:
    """Enhanced arXiv API client with advanced features."""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.cache = ArxivCache()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # arXiv category mappings
        self.categories = {
            "cs.AI": "Artificial Intelligence",
            "cs.LG": "Machine Learning", 
            "cs.CL": "Computation and Language",
            "cs.CV": "Computer Vision",
            "cs.NE": "Neural Networks",
            "cs.RO": "Robotics",
            "stat.ML": "Machine Learning (Statistics)",
            "math.OC": "Optimization and Control",
            "physics.comp-ph": "Computational Physics",
            "eess.AS": "Audio and Speech Processing"
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def parse_entry(self, entry) -> ArxivPaper:
        """Parse feedparser entry into ArxivPaper object."""
        # Extract arXiv ID
        arxiv_id = entry.id.split("/")[-1]
        
        # Extract PDF URL
        pdf_url = None
        for link in entry.links:
            if link.type == "application/pdf":
                pdf_url = link.href
                break
        
        # Extract categories
        categories = []
        if hasattr(entry, 'arxiv_primary_category'):
            categories.append(entry.arxiv_primary_category['term'])
        if hasattr(entry, 'tags'):
            categories.extend([tag['term'] for tag in entry.tags])
        
        # Create paper object
        paper = ArxivPaper(
            arxiv_id=arxiv_id,
            title=entry.title.strip(),
            authors=[author.name for author in entry.authors],
            abstract=entry.summary.strip(),
            published=entry.published,
            updated=entry.updated if hasattr(entry, 'updated') else entry.published,
            categories=categories,
            pdf_url=pdf_url or "",
            arxiv_url=entry.link,
            comment=getattr(entry, 'arxiv_comment', ""),
            journal_ref=getattr(entry, 'arxiv_journal_ref', ""),
            doi=getattr(entry, 'arxiv_doi', "")
        )
        
        # Enhance with estimated metadata
        paper.reading_time_minutes = self._estimate_reading_time(paper.abstract)
        paper.quality_score = self._calculate_quality_score(paper)
        
        return paper
    
    def _estimate_reading_time(self, abstract: str) -> int:
        """Estimate reading time based on abstract length."""
        words = len(abstract.split())
        # Assume 200 words per minute, scale from abstract
        estimated_paper_words = words * 20  # Abstract is roughly 1/20 of paper
        return max(5, estimated_paper_words // 200)
    
    def _calculate_quality_score(self, paper: ArxivPaper) -> float:
        """Calculate quality score based on available metadata."""
        score = 0.0
        
        # Length indicators (longer abstracts often indicate more thorough work)
        if len(paper.abstract) > 500:
            score += 0.2
        
        # Author count (collaboration indicator)
        if len(paper.authors) > 3:
            score += 0.1
        elif len(paper.authors) > 6:
            score += 0.2
        
        # Journal reference (peer-reviewed)
        if paper.journal_ref:
            score += 0.3
        
        # DOI (published work)
        if paper.doi:
            score += 0.2
        
        # Category prestige (some categories are more competitive)
        prestigious_cats = {'cs.LG', 'cs.AI', 'cs.CL', 'cs.CV'}
        if any(cat in prestigious_cats for cat in paper.categories):
            score += 0.2
        
        return min(1.0, score)
    
    async def search_papers(self, query: ArxivSearchQuery) -> List[ArxivPaper]:
        """Search arXiv with enhanced query and caching."""
        # Check cache first
        cached_results = self.cache.get_cached_search(query)
        if cached_results:
            log.info(f"Returning {len(cached_results)} cached results")
            return cached_results
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Build query parameters
        params = {
            "search_query": query.to_arxiv_query(),
            "start": 0,
            "max_results": query.max_results,
            "sortBy": "relevance" if query.sort_by == "relevance" else "lastUpdatedDate",
            "sortOrder": "descending"
        }
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    papers = []
                    for entry in feed.entries:
                        try:
                            paper = self.parse_entry(entry)
                            papers.append(paper)
                        except Exception as e:
                            log.error(f"Error parsing entry: {e}")
                            continue
                    
                    # Apply additional filtering
                    papers = self._apply_filters(papers, query)
                    
                    # Cache results
                    self.cache.cache_search_results(query, papers)
                    
                    log.info(f"Found {len(papers)} papers")
                    return papers
                else:
                    raise Exception(f"arXiv API returned status {response.status}")
        
        except Exception as e:
            log.error(f"Error searching arXiv: {e}")
            raise
    
    def _apply_filters(self, papers: List[ArxivPaper], query: ArxivSearchQuery) -> List[ArxivPaper]:
        """Apply additional client-side filters."""
        filtered = papers
        
        # Date filtering
        if query.date_from or query.date_to:
            filtered = []
            for paper in papers:
                paper_date = datetime.fromisoformat(paper.published.replace('Z', '+00:00'))
                
                if query.date_from and paper_date < query.date_from:
                    continue
                if query.date_to and paper_date > query.date_to:
                    continue
                
                filtered.append(paper)
        
        # Author filtering (more precise than API)
        if query.authors:
            author_filtered = []
            for paper in filtered:
                paper_authors = [author.lower() for author in paper.authors]
                if any(query_author.lower() in " ".join(paper_authors) 
                       for query_author in query.authors):
                    author_filtered.append(paper)
            filtered = author_filtered
        
        # Sort results
        if query.sort_by == "recent":
            filtered.sort(key=lambda p: p.published, reverse=True)
        elif query.sort_by == "citations":
            filtered.sort(key=lambda p: p.citation_count, reverse=True)
        
        return filtered
    
    async def get_paper_details(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Get detailed information for a specific paper."""
        # Check cache first
        cached_paper = self.cache.get_paper(arxiv_id)
        if cached_paper:
            return cached_paper
        
        # Search for the specific paper
        query = ArxivSearchQuery(text_query=f"id:{arxiv_id}", max_results=1)
        papers = await self.search_papers(query)
        
        return papers[0] if papers else None
    
    async def get_related_papers(self, paper: ArxivPaper, limit: int = 10) -> List[ArxivPaper]:
        """Find papers related to the given paper."""
        # Extract key terms from title and abstract
        title_words = paper.title.lower().split()
        abstract_words = paper.abstract.lower().split()
        
        # Find significant terms (longer than 3 chars, not common words)
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'day'}
        key_terms = []
        
        for word in title_words + abstract_words[:50]:  # First 50 words of abstract
            if len(word) > 3 and word not in common_words and word.isalpha():
                key_terms.append(word)
        
        # Use most frequent terms for search
        term_counts = defaultdict(int)
        for term in key_terms:
            term_counts[term] += 1
        
        top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        search_terms = [term for term, count in top_terms]
        
        # Search for related papers
        query = ArxivSearchQuery(
            text_query=" OR ".join(search_terms),
            categories=paper.categories[:2],  # Use main categories
            max_results=limit + 5  # Get a few extra to filter out the original
        )
        
        related = await self.search_papers(query)
        
        # Filter out the original paper and return top results
        related = [p for p in related if p.arxiv_id != paper.arxiv_id]
        return related[:limit]
    
    async def download_pdf(self, paper: ArxivPaper, download_dir: Path) -> Path:
        """Download paper PDF with progress tracking."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        filename = f"{paper.arxiv_id.replace('/', '_')}.pdf"
        download_path = download_dir / filename
        
        if download_path.exists():
            paper.is_downloaded = True
            return download_path
        
        try:
            async with self.session.get(paper.pdf_url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Update file size
                    paper.file_size = len(content)
                    
                    with open(download_path, 'wb') as f:
                        f.write(content)
                    
                    paper.is_downloaded = True
                    self.cache.update_paper(paper)
                    
                    return download_path
                else:
                    raise Exception(f"Failed to download PDF: HTTP {response.status}")
        
        except Exception as e:
            log.error(f"Error downloading PDF for {paper.arxiv_id}: {e}")
            raise


class RecommendationEngine:
    """Intelligent paper recommendation system."""
    
    def __init__(self, arxiv_api: EnhancedArxivAPI):
        self.api = arxiv_api
        self.user_interests: Dict[str, float] = defaultdict(float)
        self.interaction_history: List[Dict[str, Any]] = []
    
    def track_interaction(self, paper: ArxivPaper, action: str, weight: float = 1.0):
        """Track user interaction for learning preferences."""
        self.interaction_history.append({
            'paper_id': paper.arxiv_id,
            'action': action,  # 'view', 'select', 'download', 'read'
            'timestamp': datetime.now(),
            'categories': paper.categories,
            'authors': paper.authors,
            'weight': weight
        })
        
        # Update interest scores
        for category in paper.categories:
            self.user_interests[f"cat:{category}"] += weight
        
        for author in paper.authors[:3]:  # Top 3 authors
            self.user_interests[f"author:{author}"] += weight * 0.5
    
    async def get_trending_papers(self, categories: List[str], limit: int = 10) -> List[ArxivPaper]:
        """Get trending papers in specified categories."""
        # Recent papers (last 30 days) with high engagement
        date_from = datetime.now() - timedelta(days=30)
        
        query = ArxivSearchQuery(
            categories=categories,
            date_from=date_from,
            sort_by="recent",
            max_results=limit * 2
        )
        
        papers = await self.api.search_papers(query)
        
        # Score by recency and estimated engagement
        scored_papers = []
        for paper in papers:
            days_old = (datetime.now() - datetime.fromisoformat(paper.published.replace('Z', '+00:00'))).days
            recency_score = max(0, 30 - days_old) / 30
            engagement_score = paper.quality_score
            
            total_score = recency_score * 0.7 + engagement_score * 0.3
            scored_papers.append((paper, total_score))
        
        # Sort by score and return top results
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        return [paper for paper, score in scored_papers[:limit]]
    
    async def get_personalized_recommendations(self, limit: int = 10) -> List[ArxivPaper]:
        """Get recommendations based on user's interaction history."""
        if not self.interaction_history:
            # Fallback to general trending papers
            return await self.get_trending_papers(['cs.AI', 'cs.LG'], limit)
        
        # Analyze user preferences
        top_categories = sorted(
            [(k, v) for k, v in self.user_interests.items() if k.startswith('cat:')],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        top_authors = sorted(
            [(k, v) for k, v in self.user_interests.items() if k.startswith('author:')],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        # Build personalized query
        categories = [cat.replace('cat:', '') for cat, score in top_categories]
        authors = [author.replace('author:', '') for author, score in top_authors]
        
        query = ArxivSearchQuery(
            categories=categories,
            authors=authors,
            date_from=datetime.now() - timedelta(days=90),
            sort_by="relevance",
            max_results=limit * 2
        )
        
        papers = await self.api.search_papers(query)
        
        # Score papers based on user interests
        scored_papers = []
        for paper in papers:
            score = 0.0
            
            # Category matching
            for category in paper.categories:
                if f"cat:{category}" in self.user_interests:
                    score += self.user_interests[f"cat:{category}"] * 0.6
            
            # Author matching
            for author in paper.authors:
                if f"author:{author}" in self.user_interests:
                    score += self.user_interests[f"author:{author}"] * 0.4
            
            # Base quality score
            score += paper.quality_score * 2
            
            scored_papers.append((paper, score))
        
        # Sort and return top recommendations
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        return [paper for paper, score in scored_papers[:limit]]