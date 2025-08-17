"""
Enhanced arXiv Search Interface

Advanced UI components for arXiv paper discovery, preview, and selection
with intelligent recommendations and collection management.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from datetime import datetime, timedelta

from textual import on, log
from textual.app import ComposeResult
from textual.widgets import (
    Static, Input, Button, ListView, ListItem, Checkbox, Select, 
    Label, ProgressBar, TextArea, Collapsible, Tabs, TabPane, Tree, TreeNode,
    RadioSet, RadioButton, DateInput
)
from textual.containers import Container, Horizontal, Vertical, Grid, ScrollableContainer
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.reactive import reactive
from textual.message import Message

from .enhanced_arxiv import (
    ArxivPaper, ArxivSearchQuery, EnhancedArxivAPI, 
    RecommendationEngine, ARXIV_AVAILABLE
)

try:
    from .paper_collections import CollectionManager, CollectionManagementModal
    COLLECTIONS_AVAILABLE = True
except ImportError:
    COLLECTIONS_AVAILABLE = False

try:
    from .citation_network import CitationAnalysisModal, CitationNetworkAnalyzer
    CITATION_ANALYSIS_AVAILABLE = True
except ImportError:
    CITATION_ANALYSIS_AVAILABLE = False


class PaperCard(Container):
    """Rich paper preview card with metadata and interactions."""
    
    DEFAULT_CSS = """
    PaperCard {
        height: auto;
        margin: 0 0 1 0;
        padding: 1;
        border: solid $secondary;
        background: $surface;
    }
    
    PaperCard.selected {
        border: solid $accent;
        background: $accent 10%;
    }
    
    PaperCard:hover {
        background: $primary 5%;
    }
    
    PaperCard .paper-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    PaperCard .paper-authors {
        color: $text-muted;
        text-style: italic;
        margin-bottom: 1;
    }
    
    PaperCard .paper-metadata {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    PaperCard .paper-abstract {
        margin-bottom: 1;
        height: 3;
    }
    
    PaperCard .paper-actions {
        height: 3;
    }
    
    PaperCard .quality-indicator {
        width: 10;
    }
    
    PaperCard .quality-high {
        color: $success;
        text-style: bold;
    }
    
    PaperCard .quality-medium {
        color: $warning;
    }
    
    PaperCard .quality-low {
        color: $error;
    }
    """
    
    class PaperSelected(Message):
        """Message sent when paper selection changes."""
        def __init__(self, paper: ArxivPaper, selected: bool) -> None:
            self.paper = paper
            self.selected = selected
            super().__init__()
    
    class PaperAction(Message):
        """Message sent when paper action is triggered."""
        def __init__(self, paper: ArxivPaper, action: str) -> None:
            self.paper = paper
            self.action = action  # 'preview', 'related', 'cite', 'download'
            super().__init__()
    
    def __init__(self, paper: ArxivPaper, **kwargs):
        super().__init__(**kwargs)
        self.paper = paper
        self.is_selected = paper.is_selected
        self._abstract_expanded = False
    
    def compose(self) -> ComposeResult:
        # Title
        yield Static(self._truncate_text(self.paper.title, 80), classes="paper-title")
        
        # Authors
        authors_text = self._format_authors(self.paper.authors)
        yield Static(f"👥 {authors_text}", classes="paper-authors")
        
        # Metadata row
        with Horizontal(classes="paper-metadata"):
            yield Static(f"📅 {self._format_date(self.paper.published)}")
            yield Static(f"🏷️ {', '.join(self.paper.categories[:3])}")
            yield Static(f"📊 Q-Score: {self.paper.quality_score:.1f}", classes=self._get_quality_class())
            if self.paper.citation_count > 0:
                yield Static(f"📈 {self.paper.citation_count} citations")
        
        # Abstract (collapsible)
        abstract_text = self._truncate_text(self.paper.abstract, 200)
        if len(self.paper.abstract) > 200:
            abstract_text += "... (click to expand)"
        
        with Collapsible(title="📖 Abstract", collapsed=True, classes="paper-abstract"):
            yield Static(abstract_text, id=f"abstract-{self.paper.arxiv_id}")
        
        # Action buttons
        with Horizontal(classes="paper-actions"):
            yield Checkbox("Select", value=self.is_selected, id=f"select-{self.paper.arxiv_id}")
            yield Button("🔍 Preview", id=f"preview-{self.paper.arxiv_id}", variant="primary")
            yield Button("🔗 Related", id=f"related-{self.paper.arxiv_id}")
            yield Button("📋 Cite", id=f"cite-{self.paper.arxiv_id}")
            yield Button("📊 Citations", id=f"citations-{self.paper.arxiv_id}")
            yield Button("📥 Download", id=f"download-{self.paper.arxiv_id}", variant="success")
            
            # Quality indicator
            quality_text = self._get_quality_text()
            yield Static(quality_text, classes=f"quality-indicator {self._get_quality_class()}")
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length."""
        return text[:max_length] + "..." if len(text) > max_length else text
    
    def _format_authors(self, authors: List[str]) -> str:
        """Format author list for display."""
        if len(authors) <= 3:
            return ", ".join(authors)
        else:
            return f"{', '.join(authors[:3])} et al. ({len(authors)} total)"
    
    def _format_date(self, date_str: str) -> str:
        """Format publication date."""
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date.strftime("%Y-%m-%d")
        except:
            return date_str[:10]
    
    def _get_quality_class(self) -> str:
        """Get CSS class for quality indicator."""
        if self.paper.quality_score >= 0.7:
            return "quality-high"
        elif self.paper.quality_score >= 0.4:
            return "quality-medium"
        else:
            return "quality-low"
    
    def _get_quality_text(self) -> str:
        """Get quality indicator text."""
        score = self.paper.quality_score
        if score >= 0.8:
            return "🌟 High"
        elif score >= 0.6:
            return "⭐ Good"
        elif score >= 0.4:
            return "📊 Fair"
        else:
            return "📉 Basic"
    
    @on(Checkbox.Changed)
    def selection_changed(self, event: Checkbox.Changed) -> None:
        """Handle selection checkbox change."""
        if event.checkbox.id == f"select-{self.paper.arxiv_id}":
            self.is_selected = event.value
            self.paper.is_selected = event.value
            
            if self.is_selected:
                self.add_class("selected")
            else:
                self.remove_class("selected")
            
            self.post_message(self.PaperSelected(self.paper, self.is_selected))
    
    @on(Button.Pressed)
    def action_pressed(self, event: Button.Pressed) -> None:
        """Handle action button presses."""
        button_id = event.button.id
        paper_id = self.paper.arxiv_id
        
        if button_id == f"preview-{paper_id}":
            self.post_message(self.PaperAction(self.paper, "preview"))
        elif button_id == f"related-{paper_id}":
            self.post_message(self.PaperAction(self.paper, "related"))
        elif button_id == f"cite-{paper_id}":
            self.post_message(self.PaperAction(self.paper, "cite"))
        elif button_id == f"citations-{paper_id}":
            self.post_message(self.PaperAction(self.paper, "citations"))
        elif button_id == f"download-{paper_id}":
            self.post_message(self.PaperAction(self.paper, "download"))
    
    @on(Collapsible.Toggled)
    def abstract_toggled(self, event: Collapsible.Toggled) -> None:
        """Handle abstract expansion."""
        if not self._abstract_expanded and not event.collapsed:
            # Expand abstract on first open
            abstract_widget = self.query_one(f"#abstract-{self.paper.arxiv_id}", Static)
            abstract_widget.update(self.paper.abstract)
            self._abstract_expanded = True


class AdvancedSearchPanel(Container):
    """Advanced search interface with filters and options."""
    
    DEFAULT_CSS = """
    AdvancedSearchPanel {
        height: auto;
        border: solid $secondary;
        padding: 1;
        margin: 0 0 1 0;
    }
    
    AdvancedSearchPanel .search-section {
        margin: 0 0 1 0;
        padding: 1;
        border: solid $secondary 50%;
    }
    
    AdvancedSearchPanel .section-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    
    AdvancedSearchPanel .filter-row {
        height: 3;
        margin: 0 0 1 0;
    }
    
    AdvancedSearchPanel .category-grid {
        height: 6;
    }
    """
    
    class SearchRequested(Message):
        """Message sent when search is requested."""
        def __init__(self, query: ArxivSearchQuery) -> None:
            self.query = query
            super().__init__()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.categories_selected: Set[str] = set()
    
    def compose(self) -> ComposeResult:
        yield Static("🔍 Advanced arXiv Search", classes="section-title")
        
        # Main search query
        with Container(classes="search-section"):
            yield Static("Search Query:", classes="section-title")
            yield Input(placeholder="e.g., attention mechanisms, transformer, neural networks", id="search-query")
        
        # Categories
        with Container(classes="search-section"):
            yield Static("Categories:", classes="section-title")
            with Grid(classes="category-grid"):
                categories = [
                    ("cs.AI", "Artificial Intelligence"),
                    ("cs.LG", "Machine Learning"),
                    ("cs.CL", "Natural Language"),
                    ("cs.CV", "Computer Vision"),
                    ("cs.NE", "Neural Networks"),
                    ("cs.RO", "Robotics"),
                    ("stat.ML", "Statistics ML"),
                    ("math.OC", "Optimization")
                ]
                
                for cat_id, cat_name in categories:
                    yield Checkbox(f"{cat_id}: {cat_name}", id=f"cat-{cat_id}")
        
        # Date range
        with Container(classes="search-section"):
            yield Static("Date Range:", classes="section-title")
            with Horizontal(classes="filter-row"):
                yield Label("From:")
                yield Input(placeholder="YYYY-MM-DD", id="date-from")
                yield Label("To:")
                yield Input(placeholder="YYYY-MM-DD", id="date-to")
        
        # Authors
        with Container(classes="search-section"):
            yield Static("Authors (comma-separated):", classes="section-title")
            yield Input(placeholder="e.g., Vaswani, Attention, LeCun", id="authors-input")
        
        # Options
        with Container(classes="search-section"):
            yield Static("Options:", classes="section-title")
            with Horizontal(classes="filter-row"):
                yield Label("Sort by:")
                yield Select([
                    ("Relevance", "relevance"),
                    ("Most Recent", "recent"),
                    ("Most Cited", "citations")
                ], value="relevance", id="sort-select")
                
                yield Label("Max Results:")
                yield Select([
                    ("10", "10"),
                    ("25", "25"),
                    ("50", "50"),
                    ("100", "100")
                ], value="25", id="limit-select")
            
            yield Checkbox("Include older versions", id="include-older")
        
        # Search button
        yield Button("🚀 Search arXiv", variant="primary", id="search-btn")
    
    @on(Button.Pressed, "#search-btn")
    def search_requested(self) -> None:
        """Handle search button press."""
        # Gather search parameters
        query_text = self.query_one("#search-query", Input).value.strip()
        
        # Get selected categories
        categories = []
        for cat_id in ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "cs.RO", "stat.ML", "math.OC"]:
            checkbox = self.query_one(f"#cat-{cat_id}", Checkbox)
            if checkbox.value:
                categories.append(cat_id)
        
        # Get date range
        date_from_str = self.query_one("#date-from", Input).value.strip()
        date_to_str = self.query_one("#date-to", Input).value.strip()
        
        date_from = None
        date_to = None
        try:
            if date_from_str:
                date_from = datetime.strptime(date_from_str, "%Y-%m-%d")
            if date_to_str:
                date_to = datetime.strptime(date_to_str, "%Y-%m-%d")
        except ValueError:
            pass  # Invalid date format, ignore
        
        # Get authors
        authors_str = self.query_one("#authors-input", Input).value.strip()
        authors = [author.strip() for author in authors_str.split(",") if author.strip()]
        
        # Get options
        sort_by = self.query_one("#sort-select", Select).value
        max_results = int(self.query_one("#limit-select", Select).value)
        include_older = self.query_one("#include-older", Checkbox).value
        
        # Create query object
        query = ArxivSearchQuery(
            text_query=query_text,
            categories=categories,
            authors=authors,
            date_from=date_from,
            date_to=date_to,
            sort_by=sort_by,
            max_results=max_results,
            include_older_versions=include_older
        )
        
        self.post_message(self.SearchRequested(query))


class RecommendationsPanel(ScrollableContainer):
    """Panel showing intelligent paper recommendations."""
    
    DEFAULT_CSS = """
    RecommendationsPanel {
        height: 1fr;
        border: solid $secondary;
        padding: 1;
    }
    
    RecommendationsPanel .rec-section {
        margin: 0 0 2 0;
    }
    
    RecommendationsPanel .rec-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    
    RecommendationsPanel .rec-description {
        color: $text-muted;
        text-style: italic;
        margin-bottom: 1;
    }
    """
    
    class RecommendationSelected(Message):
        """Message sent when recommendation is selected."""
        def __init__(self, papers: List[ArxivPaper], title: str) -> None:
            self.papers = papers
            self.title = title
            super().__init__()
    
    def __init__(self, recommendation_engine: RecommendationEngine, **kwargs):
        super().__init__(**kwargs)
        self.rec_engine = recommendation_engine
        self.recommendations: Dict[str, List[ArxivPaper]] = {}
    
    def compose(self) -> ComposeResult:
        yield Static("🤖 Intelligent Recommendations", classes="rec-title")
        
        # Trending section
        with Container(classes="rec-section", id="trending-section"):
            yield Static("🔥 Trending This Week", classes="rec-title")
            yield Static("Popular papers in AI/ML from the last 7 days", classes="rec-description")
            yield Button("📊 Load Trending Papers", id="load-trending", variant="primary")
        
        # Personalized section  
        with Container(classes="rec-section", id="personalized-section"):
            yield Static("👤 Personalized for You", classes="rec-title")
            yield Static("Based on your search and selection patterns", classes="rec-description")
            yield Button("🎯 Load Personal Recommendations", id="load-personalized", variant="success")
        
        # Foundation papers section
        with Container(classes="rec-section", id="foundation-section"):
            yield Static("📚 Foundation Papers", classes="rec-title")
            yield Static("Essential papers everyone should know", classes="rec-description")
            yield Button("🏛️ Load Foundation Papers", id="load-foundation")
        
        # Recent breakthroughs section
        with Container(classes="rec-section", id="breakthroughs-section"):
            yield Static("💡 Recent Breakthroughs", classes="rec-title")
            yield Static("Highly cited papers from the last 6 months", classes="rec-description")
            yield Button("⚡ Load Breakthroughs", id="load-breakthroughs")
    
    @on(Button.Pressed)
    async def recommendation_requested(self, event: Button.Pressed) -> None:
        """Handle recommendation button presses."""
        button_id = event.button.id
        
        try:
            if button_id == "load-trending":
                papers = await self.rec_engine.get_trending_papers(['cs.AI', 'cs.LG', 'cs.CL'], 10)
                self.recommendations["trending"] = papers
                self.post_message(self.RecommendationSelected(papers, "Trending Papers"))
            
            elif button_id == "load-personalized":
                papers = await self.rec_engine.get_personalized_recommendations(10)
                self.recommendations["personalized"] = papers
                self.post_message(self.RecommendationSelected(papers, "Personalized Recommendations"))
            
            elif button_id == "load-foundation":
                # Foundation papers - search for highly cited classical papers
                foundation_queries = [
                    "attention is all you need",
                    "deep learning",
                    "neural networks",
                    "convolutional neural networks",
                    "recurrent neural networks"
                ]
                
                all_papers = []
                for query_text in foundation_queries:
                    query = ArxivSearchQuery(
                        text_query=query_text,
                        categories=['cs.AI', 'cs.LG'],
                        sort_by="citations",
                        max_results=2
                    )
                    papers = await self.rec_engine.api.search_papers(query)
                    all_papers.extend(papers)
                
                # Remove duplicates and sort by quality
                unique_papers = {p.arxiv_id: p for p in all_papers}.values()
                sorted_papers = sorted(unique_papers, key=lambda p: p.quality_score, reverse=True)[:10]
                
                self.recommendations["foundation"] = sorted_papers
                self.post_message(self.RecommendationSelected(sorted_papers, "Foundation Papers"))
            
            elif button_id == "load-breakthroughs":
                # Recent breakthroughs - highly cited recent papers
                date_from = datetime.now() - timedelta(days=180)
                query = ArxivSearchQuery(
                    categories=['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV'],
                    date_from=date_from,
                    sort_by="citations",
                    max_results=15
                )
                
                papers = await self.rec_engine.api.search_papers(query)
                # Filter for high quality papers
                breakthrough_papers = [p for p in papers if p.quality_score > 0.6][:10]
                
                self.recommendations["breakthroughs"] = breakthrough_papers
                self.post_message(self.RecommendationSelected(breakthrough_papers, "Recent Breakthroughs"))
        
        except Exception as e:
            log.error(f"Error loading recommendations: {e}")
            self.app.notify(f"Error loading recommendations: {str(e)}", severity="error")


class EnhancedArxivSearchModal(ModalScreen[Dict[str, Any]]):
    """Main modal for enhanced arXiv search and paper management."""
    
    DEFAULT_CSS = """
    EnhancedArxivSearchModal {
        align: center middle;
    }

    EnhancedArxivSearchModal > Container {
        width: 98;
        height: 50;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }

    EnhancedArxivSearchModal Tabs {
        height: 1fr;
    }

    EnhancedArxivSearchModal .search-results {
        height: 1fr;
    }

    EnhancedArxivSearchModal .results-header {
        height: 3;
        background: $primary 10%;
        padding: 1;
        margin: 0 0 1 0;
    }

    EnhancedArxivSearchModal .results-list {
        height: 1fr;
    }

    EnhancedArxivSearchModal .selection-summary {
        height: 4;
        background: $surface;
        border: solid $secondary;
        padding: 1;
        margin: 1 0;
    }

    EnhancedArxivSearchModal .action-buttons {
        height: 3;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "search", "Search"),
        Binding("ctrl+a", "select_all", "Select All"),
        Binding("ctrl+c", "clear_selection", "Clear Selection"),
        Binding("enter", "insert_selected", "Insert Selected"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arxiv_api = EnhancedArxivAPI()
        self.rec_engine = RecommendationEngine(self.arxiv_api)
        self.current_papers: List[ArxivPaper] = []
        self.selected_papers: List[ArxivPaper] = []
        self.search_history: List[ArxivSearchQuery] = []
        
        # Collection management
        if COLLECTIONS_AVAILABLE:
            self.collection_manager = CollectionManager()
        else:
            self.collection_manager = None

    def compose(self) -> ComposeResult:
        if not ARXIV_AVAILABLE:
            yield Static("❌ arXiv integration requires 'feedparser' and 'requests' packages")
            yield Button("Cancel", id="cancel-btn")
            return

        with Container():
            yield Static("🔬 Enhanced arXiv Research Assistant", classes="section-title")

            with Tabs():
                # Search tab
                with TabPane("🔍 Search", id="search-tab"):
                    yield AdvancedSearchPanel(id="search-panel")
                    
                    with Container(classes="search-results", id="search-results"):
                        # Results header
                        with Container(classes="results-header"):
                            yield Static("No search performed yet", id="results-status")
                            with Horizontal():
                                yield Button("🔄 Refresh", id="refresh-btn")
                                yield Button("📊 Select All", id="select-all-btn")
                                yield Button("🗑️ Clear Selection", id="clear-selection-btn")
                        
                        # Results list
                        yield ScrollableContainer(id="results-container", classes="results-list")

                # Recommendations tab
                with TabPane("🤖 Recommendations", id="recommendations-tab"):
                    yield RecommendationsPanel(self.rec_engine, id="recommendations-panel")

                # Collections tab
                with TabPane("📚 Collections", id="collections-tab"):
                    if COLLECTIONS_AVAILABLE:
                        yield Static("📚 Research Collections", classes="section-title")
                        yield Static("Organize papers into custom research collections for better knowledge management")
                        with Horizontal():
                            yield Button("📂 Manage Collections", id="manage-collections-btn", variant="primary")
                            yield Button("➕ Quick Add to Collection", id="quick-add-collection-btn", disabled=True)
                        yield Static("🚧 Collection browsing interface coming soon!")
                    else:
                        yield Static("🚧 Collections feature requires additional modules!")
                        yield Static("Install the collections module to organize papers")

            # Selection summary
            with Container(classes="selection-summary"):
                yield Static("📊 Selection Summary", classes="section-title")
                yield Static("No papers selected", id="selection-summary")
                yield ProgressBar(id="download-progress", show_eta=False, show_percentage=True)

            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("🚀 Insert Selected Papers", variant="primary", id="insert-btn", disabled=True)
                yield Button("📥 Download All", variant="success", id="download-all-btn", disabled=True)
                if CITATION_ANALYSIS_AVAILABLE:
                    yield Button("📊 Citation Analysis", id="citation-analysis-btn", disabled=True)
                yield Button("❌ Cancel", variant="default", id="cancel-btn")

    async def on_mount(self) -> None:
        """Initialize the modal."""
        if ARXIV_AVAILABLE:
            await self.arxiv_api.__aenter__()
            # Load some initial trending papers
            try:
                trending = await self.rec_engine.get_trending_papers(['cs.AI', 'cs.LG'], 5)
                if trending:
                    await self.display_papers(trending, "🔥 Trending Papers (AI/ML)")
            except Exception as e:
                log.error(f"Failed to load initial recommendations: {e}")

    async def on_unmount(self) -> None:
        """Cleanup when modal is closed."""
        if hasattr(self.arxiv_api, '__aexit__'):
            await self.arxiv_api.__aexit__(None, None, None)

    @on(AdvancedSearchPanel.SearchRequested)
    async def handle_search(self, event: AdvancedSearchPanel.SearchRequested) -> None:
        """Handle search request."""
        try:
            # Update status
            status = self.query_one("#results-status", Static)
            status.update("🔍 Searching arXiv...")

            # Perform search
            papers = await self.arxiv_api.search_papers(event.query)
            
            # Track search for recommendations
            self.search_history.append(event.query)
            
            # Display results
            if papers:
                await self.display_papers(papers, f"📄 Search Results ({len(papers)} papers)")
                status.update(f"✅ Found {len(papers)} papers")
            else:
                await self.display_papers([], "❌ No papers found")
                status.update("❌ No papers found")

        except Exception as e:
            log.error(f"Search error: {e}")
            status = self.query_one("#results-status", Static)
            status.update(f"❌ Search failed: {str(e)}")

    @on(RecommendationsPanel.RecommendationSelected)
    async def handle_recommendation(self, event: RecommendationsPanel.RecommendationSelected) -> None:
        """Handle recommendation selection."""
        await self.display_papers(event.papers, f"🤖 {event.title}")
        
        # Switch to search tab to show results
        tabs = self.query_one(Tabs)
        tabs.active = "search-tab"

    async def display_papers(self, papers: List[ArxivPaper], title: str) -> None:
        """Display papers in the results container."""
        self.current_papers = papers
        
        results_container = self.query_one("#results-container", ScrollableContainer)
        await results_container.remove_children()
        
        if not papers:
            await results_container.mount(Static("No papers to display"))
            return
        
        # Add paper cards
        for paper in papers:
            card = PaperCard(paper)
            await results_container.mount(card)
        
        # Update status
        status = self.query_one("#results-status", Static)
        status.update(title)

    @on(PaperCard.PaperSelected)
    def handle_paper_selection(self, event: PaperCard.PaperSelected) -> None:
        """Handle paper selection changes."""
        if event.selected:
            if event.paper not in self.selected_papers:
                self.selected_papers.append(event.paper)
                # Track selection for recommendations
                self.rec_engine.track_interaction(event.paper, 'select', 1.0)
        else:
            if event.paper in self.selected_papers:
                self.selected_papers.remove(event.paper)
        
        self.update_selection_summary()

    @on(PaperCard.PaperAction)
    async def handle_paper_action(self, event: PaperCard.PaperAction) -> None:
        """Handle paper actions."""
        if event.action == "preview":
            await self.show_paper_preview(event.paper)
        elif event.action == "related":
            await self.show_related_papers(event.paper)
        elif event.action == "cite":
            self.show_citation(event.paper)
        elif event.action == "citations":
            await self.show_citation_analysis(event.paper)
        elif event.action == "download":
            await self.download_paper(event.paper)

    async def show_paper_preview(self, paper: ArxivPaper) -> None:
        """Show detailed paper preview."""
        # Track preview interaction
        self.rec_engine.track_interaction(paper, 'view', 0.5)
        
        # Create preview modal (simplified for now)
        preview_text = f"""
📄 {paper.title}

👥 Authors: {', '.join(paper.authors)}
📅 Published: {paper.published}
🏷️ Categories: {', '.join(paper.categories)}
📊 Quality Score: {paper.quality_score:.2f}

📖 Abstract:
{paper.abstract}

🔗 arXiv ID: {paper.arxiv_id}
🔗 PDF URL: {paper.pdf_url}
"""
        self.app.notify(preview_text, timeout=15)

    async def show_related_papers(self, paper: ArxivPaper) -> None:
        """Show papers related to the selected paper."""
        try:
            related_papers = await self.arxiv_api.get_related_papers(paper, limit=10)
            await self.display_papers(related_papers, f"🔗 Papers Related to: {paper.title[:50]}...")
        except Exception as e:
            log.error(f"Error finding related papers: {e}")
            self.app.notify(f"Error finding related papers: {str(e)}", severity="error")

    def show_citation(self, paper: ArxivPaper) -> None:
        """Show citation information."""
        citation = f"""
@article{{{paper.arxiv_id.replace('/', '_')},
    title={{{paper.title}}},
    author={{{' and '.join(paper.authors)}}},
    journal={{arXiv preprint arXiv:{paper.arxiv_id}}},
    year={{{paper.published[:4]}}},
    url={{{paper.arxiv_url}}}
}}
"""
        self.app.notify(f"BibTeX Citation:\n{citation}", timeout=10)

    async def show_citation_analysis(self, paper: ArxivPaper) -> None:
        """Show citation network analysis for a single paper."""
        if not CITATION_ANALYSIS_AVAILABLE:
            self.app.notify("Citation analysis not available", severity="error")
            return

        def handle_analysis_result(result):
            if result:
                # Show analysis report
                self.app.notify("Citation analysis completed", severity="success")

        self.app.push_screen(CitationAnalysisModal([paper], self.arxiv_api), handle_analysis_result)

    async def download_paper(self, paper: ArxivPaper) -> None:
        """Download individual paper."""
        try:
            # Create temp directory
            download_dir = Path(tempfile.mkdtemp(prefix="elia_arxiv_single_"))
            
            # Download paper
            downloaded_path = await self.arxiv_api.download_pdf(paper, download_dir)
            
            # Track download interaction
            self.rec_engine.track_interaction(paper, 'download', 1.5)
            
            self.app.notify(f"✅ Downloaded: {paper.title[:50]}... to {downloaded_path}", timeout=5)
            
        except Exception as e:
            log.error(f"Download error: {e}")
            self.app.notify(f"❌ Download failed: {str(e)}", severity="error")

    def update_selection_summary(self) -> None:
        """Update the selection summary display."""
        count = len(self.selected_papers)
        summary = self.query_one("#selection-summary", Static)
        insert_btn = self.query_one("#insert-btn", Button)
        download_btn = self.query_one("#download-all-btn", Button)
        
        # Enable quick add to collection button if collections are available
        if COLLECTIONS_AVAILABLE:
            try:
                quick_add_btn = self.query_one("#quick-add-collection-btn", Button)
                quick_add_btn.disabled = count == 0
            except:
                pass  # Button might not exist in current tab
        
        # Enable citation analysis button if available
        if CITATION_ANALYSIS_AVAILABLE:
            try:
                citation_btn = self.query_one("#citation-analysis-btn", Button)
                citation_btn.disabled = count == 0
            except:
                pass  # Button might not exist
        
        if count == 0:
            summary.update("No papers selected")
            insert_btn.disabled = True
            download_btn.disabled = True
        else:
            # Calculate total estimated reading time
            total_time = sum(paper.reading_time_minutes for paper in self.selected_papers)
            avg_quality = sum(paper.quality_score for paper in self.selected_papers) / count
            
            summary.update(
                f"📊 {count} papers selected • "
                f"⏱️ ~{total_time} min reading time • "
                f"📈 Avg quality: {avg_quality:.2f}"
            )
            insert_btn.disabled = False
            download_btn.disabled = False

    @on(Button.Pressed, "#select-all-btn")
    def select_all_papers(self) -> None:
        """Select all visible papers."""
        results_container = self.query_one("#results-container", ScrollableContainer)
        
        for child in results_container.children:
            if isinstance(child, PaperCard):
                checkbox = child.query_one(f"#select-{child.paper.arxiv_id}", Checkbox)
                if not checkbox.value:
                    checkbox.value = True
                    child.is_selected = True
                    child.paper.is_selected = True
                    child.add_class("selected")
                    
                    if child.paper not in self.selected_papers:
                        self.selected_papers.append(child.paper)
        
        self.update_selection_summary()

    @on(Button.Pressed, "#clear-selection-btn")
    def clear_all_selections(self) -> None:
        """Clear all selections."""
        results_container = self.query_one("#results-container", ScrollableContainer)
        
        for child in results_container.children:
            if isinstance(child, PaperCard):
                checkbox = child.query_one(f"#select-{child.paper.arxiv_id}", Checkbox)
                checkbox.value = False
                child.is_selected = False
                child.paper.is_selected = False
                child.remove_class("selected")
        
        self.selected_papers.clear()
        self.update_selection_summary()

    @on(Button.Pressed, "#insert-btn")
    async def insert_selected_papers(self) -> None:
        """Insert selected papers."""
        if not self.selected_papers:
            self.app.notify("No papers selected", severity="warning")
            return

        try:
            # Show progress
            progress = self.query_one("#download-progress", ProgressBar)
            progress.total = len(self.selected_papers)
            progress.progress = 0
            
            # Create temp directory for downloads
            download_dir = Path(tempfile.mkdtemp(prefix="elia_arxiv_"))
            downloaded_files = []
            
            # Download selected papers
            for i, paper in enumerate(self.selected_papers):
                try:
                    downloaded_path = await self.arxiv_api.download_pdf(paper, download_dir)
                    downloaded_files.append(downloaded_path)
                    progress.progress = i + 1
                    
                    # Track download for recommendations
                    self.rec_engine.track_interaction(paper, 'download', 2.0)
                    
                except Exception as e:
                    log.error(f"Failed to download {paper.arxiv_id}: {e}")
            
            # Prepare result data
            result = {
                "mode": "enhanced_arxiv",
                "files": downloaded_files,
                "papers": self.selected_papers,
                "options": {
                    "skip_duplicates": True,
                    "recursive": False,
                    "file_type_filter": "pdf"
                },
                "download_dir": download_dir,
                "total_files": len(downloaded_files)
            }
            
            self.dismiss(result)
            
        except Exception as e:
            log.error(f"Error inserting papers: {e}")
            self.app.notify(f"Error: {str(e)}", severity="error")

    @on(Button.Pressed, "#manage-collections-btn")
    def manage_collections(self) -> None:
        """Open collection management modal."""
        if not COLLECTIONS_AVAILABLE or not self.collection_manager:
            self.app.notify("Collections feature not available", severity="error")
            return
        
        def handle_collection_result(result):
            if result:
                # Collection was created/modified
                self.app.notify(f"Collection updated successfully", severity="success")
        
        self.app.push_screen(CollectionManagementModal(self.collection_manager), handle_collection_result)

    @on(Button.Pressed, "#quick-add-collection-btn")
    def quick_add_to_collection(self) -> None:
        """Quick add selected papers to a collection."""
        if not self.selected_papers:
            self.app.notify("No papers selected", severity="warning")
            return
        
        if not COLLECTIONS_AVAILABLE or not self.collection_manager:
            self.app.notify("Collections feature not available", severity="error")
            return
        
        # Simple implementation: add to a default "Quick Saves" collection
        quick_collection = None
        for collection in self.collection_manager.collections.values():
            if collection.name == "Quick Saves":
                quick_collection = collection
                break
        
        if not quick_collection:
            quick_collection = self.collection_manager.create_collection(
                "Quick Saves", 
                "Quickly saved papers from arXiv search"
            )
        
        # Add selected papers
        added_count = 0
        for paper in self.selected_papers:
            if paper.arxiv_id not in quick_collection.papers:
                self.collection_manager.add_paper_to_collection(quick_collection.id, paper.arxiv_id)
                added_count += 1
        
        self.app.notify(f"Added {added_count} papers to 'Quick Saves' collection", severity="success")

    @on(Button.Pressed, "#citation-analysis-btn")
    def analyze_selected_citations(self) -> None:
        """Analyze citations for selected papers."""
        if not self.selected_papers:
            self.app.notify("No papers selected", severity="warning")
            return
        
        if not CITATION_ANALYSIS_AVAILABLE:
            self.app.notify("Citation analysis not available", severity="error")
            return

        def handle_analysis_result(result):
            if result:
                # Show analysis report
                self.app.notify("Citation analysis report generated", severity="success")

        self.app.push_screen(CitationAnalysisModal(self.selected_papers, self.arxiv_api), handle_analysis_result)

    @on(Button.Pressed, "#cancel-btn")
    def cancel_search(self) -> None:
        """Cancel the search."""
        self.dismiss(None)

    # Action handlers for key bindings
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss(None)

    def action_search(self) -> None:
        """Trigger search via keyboard."""
        search_panel = self.query_one("#search-panel", AdvancedSearchPanel)
        search_panel.search_requested()

    def action_select_all(self) -> None:
        """Select all papers via keyboard."""
        self.select_all_papers()

    def action_clear_selection(self) -> None:
        """Clear selection via keyboard."""
        self.clear_all_selections()

    def action_insert_selected(self) -> None:
        """Insert selected papers via keyboard."""
        if self.selected_papers:
            self.run_worker(self.insert_selected_papers(), exclusive=True)