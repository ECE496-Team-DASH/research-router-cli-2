"""
Simple arXiv Paper Downloader

Just downloads papers to a session folder without automatic insertion.
Clean interface, no emoji symbols, terminal-friendly.
"""

import asyncio
import aiohttp
import tempfile
from pathlib import Path
from typing import List, Dict, Any

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False


async def search_arxiv_papers(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search arXiv for papers."""
    if not FEEDPARSER_AVAILABLE:
        raise ImportError("feedparser is required. Install with: pip install feedparser")
    
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    
    papers = []
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            if response.status == 200:
                content = await response.text()
                feed = feedparser.parse(content)
                
                for entry in feed.entries:
                    # Extract PDF URL
                    pdf_url = None
                    for link in entry.links:
                        if link.type == "application/pdf":
                            pdf_url = link.href
                            break
                    
                    if pdf_url:
                        # Clean up title and authors
                        title = entry.title.strip()
                        authors = [author.name for author in entry.authors]
                        authors_str = ", ".join(authors[:3])  # Show first 3 authors
                        if len(authors) > 3:
                            authors_str += " et al."
                        
                        papers.append({
                            "title": title,
                            "authors": authors,
                            "authors_str": authors_str,
                            "pdf_url": pdf_url,
                            "arxiv_id": entry.id.split("/")[-1],
                            "published": getattr(entry, 'published', 'Unknown'),
                            "summary": getattr(entry, 'summary', '')[:200] + "..."
                        })
    
    return papers


async def download_selected_papers(papers: List[Dict[str, Any]], selected_indices: List[int], 
                                 download_dir: Path) -> Dict[str, Any]:
    """Download selected papers to a directory."""
    download_dir.mkdir(parents=True, exist_ok=True)
    
    selected_papers = [papers[i] for i in selected_indices if 0 <= i < len(papers)]
    
    successful = 0
    failed = 0
    downloaded_files = []
    
    async with aiohttp.ClientSession() as session:
        for i, paper in enumerate(selected_papers, 1):
            try:
                filename = f"{paper['arxiv_id'].replace('/', '_')}.pdf"
                download_path = download_dir / filename
                
                # Skip if already exists
                if download_path.exists():
                    successful += 1
                    downloaded_files.append(str(download_path))
                    continue
                
                async with session.get(paper['pdf_url']) as pdf_response:
                    if pdf_response.status == 200:
                        content = await pdf_response.read()
                        with open(download_path, 'wb') as f:
                            f.write(content)
                        
                        successful += 1
                        downloaded_files.append(str(download_path))
                    else:
                        failed += 1
            
            except Exception:
                failed += 1
    
    return {
        "successful": successful,
        "failed": failed,
        "downloaded_files": downloaded_files,
        "download_dir": str(download_dir)
    }