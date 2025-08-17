"""
Citation Network Analysis for arXiv Papers

Provides citation relationship analysis, academic lineage tracking,
and research impact visualization for enhanced paper discovery.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime

from textual import on, log
from textual.app import ComposeResult
from textual.widgets import Static, Tree, TreeNode, ListView, ListItem, Button, Input
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.message import Message

try:
    from .enhanced_arxiv import ArxivPaper, EnhancedArxivAPI
    ARXIV_INTEGRATION = True
except ImportError:
    ARXIV_INTEGRATION = False


@dataclass
class CitationRelationship:
    """Represents a citation relationship between papers."""
    citing_paper: str  # arXiv ID of citing paper
    cited_paper: str   # arXiv ID of cited paper
    citation_type: str = "reference"  # reference, self_citation, etc.
    confidence: float = 1.0  # Confidence in the relationship
    context: str = ""  # Context where citation appears


@dataclass
class PaperMetrics:
    """Comprehensive metrics for a research paper."""
    arxiv_id: str
    title: str
    
    # Citation metrics
    citation_count: int = 0
    h_index_contribution: float = 0.0
    impact_factor: float = 0.0
    
    # Network metrics
    in_degree: int = 0  # Papers citing this one
    out_degree: int = 0  # Papers this one cites
    betweenness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    
    # Temporal metrics
    citation_velocity: float = 0.0  # Citations per month
    peak_citation_period: Optional[str] = None
    
    # Research lineage
    academic_ancestors: List[str] = field(default_factory=list)  # Foundational papers
    academic_descendants: List[str] = field(default_factory=list)  # Papers building on this
    
    # Quality indicators
    novelty_score: float = 0.0
    influence_score: float = 0.0
    interdisciplinary_score: float = 0.0


class CitationNetwork:
    """Manages citation relationships and network analysis."""
    
    def __init__(self):
        self.relationships: List[CitationRelationship] = []
        self.papers: Dict[str, PaperMetrics] = {}
        self.citation_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_citation_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Caching for expensive computations
        self._centrality_cache: Dict[str, Dict[str, float]] = {}
        self._path_cache: Dict[Tuple[str, str], List[str]] = {}
    
    def add_paper(self, paper: ArxivPaper) -> PaperMetrics:
        """Add a paper to the network."""
        if paper.arxiv_id not in self.papers:
            metrics = PaperMetrics(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                citation_count=paper.citation_count
            )
            self.papers[paper.arxiv_id] = metrics
        
        return self.papers[paper.arxiv_id]
    
    def add_citation_relationship(self, citing_id: str, cited_id: str, 
                                context: str = "", confidence: float = 1.0):
        """Add a citation relationship."""
        relationship = CitationRelationship(
            citing_paper=citing_id,
            cited_paper=cited_id,
            context=context,
            confidence=confidence
        )
        
        self.relationships.append(relationship)
        self.citation_graph[citing_id].add(cited_id)
        self.reverse_citation_graph[cited_id].add(citing_id)
        
        # Update metrics
        if citing_id in self.papers:
            self.papers[citing_id].out_degree += 1
        if cited_id in self.papers:
            self.papers[cited_id].in_degree += 1
            self.papers[cited_id].citation_count += 1
        
        # Clear caches
        self._centrality_cache.clear()
        self._path_cache.clear()
    
    def get_direct_citations(self, arxiv_id: str) -> List[str]:
        """Get papers directly citing this paper."""
        return list(self.reverse_citation_graph.get(arxiv_id, set()))
    
    def get_direct_references(self, arxiv_id: str) -> List[str]:
        """Get papers directly referenced by this paper."""
        return list(self.citation_graph.get(arxiv_id, set()))
    
    def find_citation_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find citation path between two papers using BFS."""
        if (source_id, target_id) in self._path_cache:
            return self._path_cache[(source_id, target_id)]
        
        if source_id == target_id:
            return [source_id]
        
        # BFS to find shortest path
        queue = [(source_id, [source_id])]
        visited = {source_id}
        
        while queue:
            current, path = queue.pop(0)
            
            # Check both directions (citations and references)
            neighbors = (self.citation_graph.get(current, set()) | 
                        self.reverse_citation_graph.get(current, set()))
            
            for neighbor in neighbors:
                if neighbor == target_id:
                    result = path + [neighbor]
                    self._path_cache[(source_id, target_id)] = result
                    return result
                
                if neighbor not in visited and len(path) < 6:  # Limit depth
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        self._path_cache[(source_id, target_id)] = None
        return None
    
    def calculate_research_lineage(self, arxiv_id: str, max_depth: int = 3) -> Dict[str, List[str]]:
        """Calculate academic ancestors and descendants."""
        if arxiv_id not in self.papers:
            return {"ancestors": [], "descendants": []}
        
        # Find ancestors (papers this work builds upon)
        ancestors = []
        current_level = {arxiv_id}
        visited = {arxiv_id}
        
        for depth in range(max_depth):
            next_level = set()
            for paper_id in current_level:
                references = self.citation_graph.get(paper_id, set())
                for ref in references:
                    if ref not in visited:
                        ancestors.append(ref)
                        next_level.add(ref)
                        visited.add(ref)
            current_level = next_level
            if not current_level:
                break
        
        # Find descendants (papers building on this work)
        descendants = []
        current_level = {arxiv_id}
        visited = {arxiv_id}
        
        for depth in range(max_depth):
            next_level = set()
            for paper_id in current_level:
                citations = self.reverse_citation_graph.get(paper_id, set())
                for cit in citations:
                    if cit not in visited:
                        descendants.append(cit)
                        next_level.add(cit)
                        visited.add(cit)
            current_level = next_level
            if not current_level:
                break
        
        return {
            "ancestors": ancestors[:20],  # Limit results
            "descendants": descendants[:20]
        }
    
    def calculate_impact_metrics(self, arxiv_id: str) -> Dict[str, float]:
        """Calculate various impact metrics for a paper."""
        if arxiv_id not in self.papers:
            return {}
        
        paper = self.papers[arxiv_id]
        citations = self.get_direct_citations(arxiv_id)
        
        # Basic impact
        citation_count = len(citations)
        
        # Weighted impact (citations from highly cited papers count more)
        weighted_impact = 0.0
        for citing_paper in citations:
            citing_citations = len(self.get_direct_citations(citing_paper))
            weight = 1.0 + (citing_citations / 100.0)  # Logarithmic weighting
            weighted_impact += weight
        
        # Novelty score (based on reference diversity)
        references = self.get_direct_references(arxiv_id)
        if references:
            # Calculate diversity of referenced paper categories/years
            novelty_score = min(1.0, len(references) / 50.0)  # More references = more novel
        else:
            novelty_score = 0.5  # Default for papers with no references
        
        # Influence score (how much this paper influences future work)
        descendants = self.calculate_research_lineage(arxiv_id, 2)["descendants"]
        influence_score = min(1.0, len(descendants) / 20.0)
        
        return {
            "citation_count": float(citation_count),
            "weighted_impact": weighted_impact,
            "novelty_score": novelty_score,
            "influence_score": influence_score,
            "network_centrality": self._calculate_centrality(arxiv_id)
        }
    
    def _calculate_centrality(self, arxiv_id: str) -> float:
        """Calculate network centrality for a paper."""
        # Simplified centrality calculation
        citations = len(self.get_direct_citations(arxiv_id))
        references = len(self.get_direct_references(arxiv_id))
        
        # Papers with many citations and references are more central
        centrality = (citations + references) / (2.0 * max(1, len(self.papers)))
        return min(1.0, centrality)
    
    def find_research_clusters(self, min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """Identify clusters of related papers."""
        # Use simple connected components for clustering
        visited = set()
        clusters = []
        
        for paper_id in self.papers:
            if paper_id in visited:
                continue
            
            # DFS to find connected component
            cluster = []
            stack = [paper_id]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                
                visited.add(current)
                cluster.append(current)
                
                # Add neighbors (both citations and references)
                neighbors = (self.citation_graph.get(current, set()) | 
                           self.reverse_citation_graph.get(current, set()))
                
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor in self.papers:
                        stack.append(neighbor)
            
            if len(cluster) >= min_cluster_size:
                # Calculate cluster metrics
                total_citations = sum(self.papers[pid].citation_count for pid in cluster)
                avg_citations = total_citations / len(cluster)
                
                clusters.append({
                    "papers": cluster,
                    "size": len(cluster),
                    "total_citations": total_citations,
                    "avg_citations": avg_citations,
                    "representative_paper": max(cluster, key=lambda p: self.papers[p].citation_count)
                })
        
        # Sort clusters by importance
        clusters.sort(key=lambda c: c["total_citations"], reverse=True)
        return clusters


class CitationNetworkAnalyzer:
    """Provides citation analysis functionality with external data integration."""
    
    def __init__(self, arxiv_api: Optional[EnhancedArxivAPI] = None):
        self.arxiv_api = arxiv_api
        self.network = CitationNetwork()
        self.semantic_scholar_cache: Dict[str, Dict] = {}
    
    async def analyze_paper_citations(self, paper: ArxivPaper) -> Dict[str, Any]:
        """Comprehensive citation analysis for a paper."""
        # Add paper to network
        metrics = self.network.add_paper(paper)
        
        # Try to get citation data from external sources
        await self._fetch_citation_data(paper.arxiv_id)
        
        # Calculate metrics
        impact_metrics = self.network.calculate_impact_metrics(paper.arxiv_id)
        lineage = self.network.calculate_research_lineage(paper.arxiv_id)
        
        return {
            "paper_id": paper.arxiv_id,
            "title": paper.title,
            "metrics": impact_metrics,
            "lineage": lineage,
            "direct_citations": self.network.get_direct_citations(paper.arxiv_id),
            "direct_references": self.network.get_direct_references(paper.arxiv_id)
        }
    
    async def _fetch_citation_data(self, arxiv_id: str):
        """Fetch citation data from external sources."""
        # This would integrate with APIs like Semantic Scholar, Google Scholar, etc.
        # For now, we'll use mock data based on the paper's metadata
        
        # Extract references from paper text (simplified)
        if self.arxiv_api:
            try:
                paper = await self.arxiv_api.get_paper_details(arxiv_id)
                if paper:
                    # Mock reference extraction from abstract
                    references = self._extract_mock_references(paper.abstract)
                    for ref_id in references:
                        if ref_id != arxiv_id:  # Avoid self-citation
                            self.network.add_citation_relationship(arxiv_id, ref_id)
            except Exception as e:
                log.debug(f"Could not fetch citation data for {arxiv_id}: {e}")
    
    def _extract_mock_references(self, text: str) -> List[str]:
        """Extract mock arXiv references from text."""
        # Simple pattern matching for arXiv IDs in text
        arxiv_pattern = r'(?:arXiv:)?(\d{4}\.\d{4,5})'
        matches = re.findall(arxiv_pattern, text)
        
        # Return first few matches as mock references
        return [f"{match}" for match in matches[:3]]
    
    async def find_influential_papers(self, papers: List[ArxivPaper], 
                                    top_k: int = 10) -> List[Dict[str, Any]]:
        """Find the most influential papers in a collection."""
        results = []
        
        for paper in papers:
            analysis = await self.analyze_paper_citations(paper)
            
            # Calculate combined influence score
            metrics = analysis["metrics"]
            influence_score = (
                metrics.get("citation_count", 0) * 0.4 +
                metrics.get("weighted_impact", 0) * 0.3 +
                metrics.get("influence_score", 0) * 0.2 +
                metrics.get("network_centrality", 0) * 0.1
            )
            
            results.append({
                "paper": paper,
                "analysis": analysis,
                "influence_score": influence_score
            })
        
        # Sort by influence score
        results.sort(key=lambda x: x["influence_score"], reverse=True)
        return results[:top_k]
    
    def generate_citation_report(self, papers: List[ArxivPaper]) -> str:
        """Generate a comprehensive citation analysis report."""
        if not papers:
            return "No papers to analyze."
        
        report_lines = []
        report_lines.append("# Citation Network Analysis Report")
        report_lines.append(f"Analysis of {len(papers)} papers\n")
        
        # Overall statistics
        total_citations = sum(p.citation_count for p in papers)
        avg_citations = total_citations / len(papers) if papers else 0
        
        report_lines.append("## Overall Statistics")
        report_lines.append(f"- Total papers: {len(papers)}")
        report_lines.append(f"- Total citations: {total_citations}")
        report_lines.append(f"- Average citations per paper: {avg_citations:.1f}")
        
        # Find clusters
        clusters = self.network.find_research_clusters()
        if clusters:
            report_lines.append(f"- Research clusters identified: {len(clusters)}")
            report_lines.append(f"- Largest cluster: {clusters[0]['size']} papers")
        
        report_lines.append("")
        
        # Top papers by citations
        sorted_papers = sorted(papers, key=lambda p: p.citation_count, reverse=True)
        report_lines.append("## Most Cited Papers")
        for i, paper in enumerate(sorted_papers[:5], 1):
            report_lines.append(f"{i}. {paper.title} ({paper.citation_count} citations)")
        
        report_lines.append("")
        
        # Research clusters
        if clusters:
            report_lines.append("## Research Clusters")
            for i, cluster in enumerate(clusters[:3], 1):
                rep_paper_id = cluster["representative_paper"]
                rep_paper = next((p for p in papers if p.arxiv_id == rep_paper_id), None)
                rep_title = rep_paper.title if rep_paper else "Unknown"
                
                report_lines.append(f"### Cluster {i}: {cluster['size']} papers")
                report_lines.append(f"Representative paper: {rep_title}")
                report_lines.append(f"Total cluster citations: {cluster['total_citations']}")
                report_lines.append("")
        
        return "\n".join(report_lines)


class CitationAnalysisModal(ModalScreen[Optional[str]]):
    """Modal for citation network analysis and visualization."""
    
    DEFAULT_CSS = """
    CitationAnalysisModal {
        align: center middle;
    }

    CitationAnalysisModal > Container {
        width: 85;
        height: 40;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }

    CitationAnalysisModal .analysis-section {
        margin: 1 0;
        padding: 1;
        border: solid $secondary;
    }

    CitationAnalysisModal .section-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    CitationAnalysisModal .report-area {
        height: 20;
        border: solid $secondary;
        padding: 1;
    }

    CitationAnalysisModal .metrics-grid {
        height: 10;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+r", "generate_report", "Generate Report"),
        Binding("ctrl+e", "export_data", "Export Data"),
    ]

    def __init__(self, papers: List[ArxivPaper], arxiv_api: Optional[EnhancedArxivAPI] = None, **kwargs):
        super().__init__(**kwargs)
        self.papers = papers
        self.analyzer = CitationNetworkAnalyzer(arxiv_api)
        self.analysis_results: Dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("📊 Citation Network Analysis", classes="section-title")

            # Analysis controls
            with Container(classes="analysis-section"):
                yield Static("Analysis Options:", classes="section-title")
                with Horizontal():
                    yield Button("🔍 Analyze Citations", id="analyze-btn", variant="primary")
                    yield Button("🏆 Find Influential", id="influential-btn")
                    yield Button("📈 Generate Report", id="report-btn")
                    yield Button("📤 Export Data", id="export-btn")

            # Results display
            with Container(classes="analysis-section"):
                yield Static("Analysis Results:", classes="section-title")
                yield ScrollableContainer(
                    Static(f"Ready to analyze {len(self.papers)} papers..."),
                    classes="report-area",
                    id="results-area"
                )

            # Action buttons
            with Horizontal():
                yield Button("❌ Close", variant="default", id="close-btn")

    @on(Button.Pressed, "#analyze-btn")
    async def analyze_citations(self) -> None:
        """Analyze citation relationships."""
        results_area = self.query_one("#results-area", ScrollableContainer)
        await results_area.remove_children()
        await results_area.mount(Static("🔍 Analyzing citation relationships..."))

        try:
            # Analyze each paper
            for i, paper in enumerate(self.papers):
                await results_area.mount(Static(f"📄 Analyzing: {paper.title[:50]}..."))
                analysis = await self.analyzer.analyze_paper_citations(paper)
                self.analysis_results[paper.arxiv_id] = analysis

            # Display summary
            await results_area.remove_children()
            await results_area.mount(Static("✅ Citation analysis completed!"))
            await results_area.mount(Static(f"📊 Analyzed {len(self.papers)} papers"))
            
            # Show basic metrics
            total_citations = sum(len(r["direct_citations"]) for r in self.analysis_results.values())
            total_references = sum(len(r["direct_references"]) for r in self.analysis_results.values())
            
            await results_area.mount(Static(f"🔗 Total citation relationships: {total_citations}"))
            await results_area.mount(Static(f"📚 Total references: {total_references}"))

        except Exception as e:
            log.error(f"Citation analysis error: {e}")
            await results_area.mount(Static(f"❌ Analysis failed: {str(e)}"))

    @on(Button.Pressed, "#influential-btn")
    async def find_influential_papers(self) -> None:
        """Find most influential papers."""
        if not self.analysis_results:
            self.app.notify("Please run citation analysis first", severity="warning")
            return

        results_area = self.query_one("#results-area", ScrollableContainer)
        await results_area.remove_children()
        await results_area.mount(Static("🏆 Finding most influential papers..."))

        try:
            influential = await self.analyzer.find_influential_papers(self.papers, 5)
            
            await results_area.remove_children()
            await results_area.mount(Static("🏆 Most Influential Papers:"))
            
            for i, result in enumerate(influential, 1):
                paper = result["paper"]
                score = result["influence_score"]
                await results_area.mount(
                    Static(f"{i}. {paper.title} (Score: {score:.2f})")
                )

        except Exception as e:
            log.error(f"Influential papers analysis error: {e}")
            await results_area.mount(Static(f"❌ Analysis failed: {str(e)}"))

    @on(Button.Pressed, "#report-btn")
    def generate_report(self) -> None:
        """Generate comprehensive citation report."""
        if not self.analysis_results:
            self.app.notify("Please run citation analysis first", severity="warning")
            return

        report = self.analyzer.generate_citation_report(self.papers)
        self.dismiss(report)

    @on(Button.Pressed, "#export-btn")
    def export_data(self) -> None:
        """Export analysis data."""
        if not self.analysis_results:
            self.app.notify("Please run citation analysis first", severity="warning")
            return

        # Create export data
        export_data = {
            "papers": [{"arxiv_id": p.arxiv_id, "title": p.title} for p in self.papers],
            "analysis_results": self.analysis_results,
            "timestamp": str(datetime.now())
        }

        # In a full implementation, this would save to file
        self.app.notify("Citation data exported (feature in development)", severity="info")

    @on(Button.Pressed, "#close-btn")
    def close_analysis(self) -> None:
        """Close the analysis modal."""
        self.dismiss(None)

    # Action handlers
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss(None)

    def action_generate_report(self) -> None:
        """Handle report generation shortcut."""
        self.generate_report()

    def action_export_data(self) -> None:
        """Handle export shortcut."""
        self.export_data()