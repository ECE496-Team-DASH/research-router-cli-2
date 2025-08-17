"""
Enhanced Document Insertion Widget for Nano-GraphRAG

Provides an improved interface for inserting documents and folders into GraphRAG sessions,
with interactive file browsing, multi-selection, bulk operations, and arXiv integration.
"""

import os
import asyncio
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Set
from datetime import datetime, timezone

from textual import on, log, events
from textual.app import ComposeResult
from textual.widgets import (
    Static, Input, Button, DirectoryTree, ListView, ListItem, 
    Checkbox, Select, Label, ProgressBar, TextArea, Tree, DataTable
)
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.reactive import reactive
from textual.message import Message

try:
    import feedparser
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

try:
    from .enhanced_arxiv_ui import EnhancedArxivSearchModal
    ENHANCED_ARXIV_AVAILABLE = True
except ImportError:
    ENHANCED_ARXIV_AVAILABLE = False


class SelectableFileItem(ListItem):
    """A selectable file item with visual indicators."""
    
    def __init__(self, file_path: Path, is_supported: bool = True, *args, **kwargs):
        self.file_path = file_path
        self.is_supported = is_supported
        self.is_selected = False
        
        # Create display text with file type indicators
        try:
            if file_path.is_dir():
                icon = "📁"
                size_text = "folder"
            else:
                ext = file_path.suffix.lower()
                if ext == ".pdf":
                    icon = "📄"
                elif ext in [".txt", ".md"]:
                    icon = "📝"
                elif ext == ".docx":
                    icon = "📋"
                else:
                    icon = "📄" if is_supported else "❓"
                
                try:
                    size = file_path.stat().st_size
                    if size < 1024:
                        size_text = f"{size} B"
                    elif size < 1024 * 1024:
                        size_text = f"{size // 1024} KB"
                    else:
                        size_text = f"{size // (1024 * 1024)} MB"
                except Exception:
                    size_text = "unknown"
            
            display_text = f"{icon} {file_path.name}"
            if not file_path.is_dir():
                display_text += f" ({size_text})"
        except Exception:
            # Fallback for invalid paths
            display_text = f"❓ {str(file_path)}"
        
        super().__init__(Static(display_text), *args, **kwargs)
        
        # Style based on support and selection
        if not is_supported and not file_path.is_dir():
            self.add_class("unsupported")
    
    def toggle_selection(self):
        """Toggle selection state."""
        self.is_selected = not self.is_selected
        if self.is_selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")
    
    def set_selected(self, selected: bool):
        """Set selection state."""
        self.is_selected = selected
        if selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")


class InteractiveFileExplorer(Container):
    """Interactive file explorer with multi-selection capabilities."""
    
    class FileSelected(Message):
        """Message sent when file selection changes."""
        def __init__(self, selected_files: List[Path]) -> None:
            self.selected_files = selected_files
            super().__init__()
    
    current_path = reactive(Path.cwd())
    
    def __init__(self, initial_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self.current_path = initial_path or Path.cwd()
        self.selected_files: Set[Path] = set()
        self.supported_extensions = {'.txt', '.md', '.pdf', '.docx'}
    
    def compose(self) -> ComposeResult:
        with Vertical():
            # Current path and navigation
            with Horizontal(id="nav-bar"):
                yield Button("↑ Up", id="up-btn", variant="primary")
                yield Static(str(self.current_path), id="current-path")
                yield Button("🏠 Home", id="home-btn")
                yield Button("📁 CWD", id="cwd-btn")
            
            # File list
            yield ListView(id="file-list")
            
            # Selection info
            with Horizontal(id="selection-info"):
                yield Static("Selected: 0 files", id="selection-count")
                yield Button("Select All", id="select-all-btn")
                yield Button("Clear", id="clear-selection-btn")
    
    def on_mount(self) -> None:
        """Initialize the file explorer."""
        self.refresh_file_list()
    
    def watch_current_path(self, new_path: Path) -> None:
        """Update display when current path changes."""
        try:
            if new_path.exists() and new_path.is_dir():
                self.query_one("#current-path", Static).update(str(new_path))
                self.refresh_file_list()
                self.selected_files.clear()
                self.update_selection_display()
        except Exception as e:
            log.error(f"Error accessing path {new_path}: {e}")
    
    def refresh_file_list(self) -> None:
        """Refresh the file list for current directory."""
        file_list = self.query_one("#file-list", ListView)
        file_list.clear()
        
        try:
            if not self.current_path.exists() or not self.current_path.is_dir():
                file_list.append(ListItem(Static("❌ Invalid directory")))
                return
            
            # Get all items in directory
            items = []
            
            # Add directories first
            for item in sorted(self.current_path.iterdir()):
                if item.is_dir() and not item.name.startswith('.'):
                    items.append(item)
            
            # Add files
            for item in sorted(self.current_path.iterdir()):
                if item.is_file() and not item.name.startswith('.'):
                    items.append(item)
            
            if not items:
                file_list.append(ListItem(Static("📂 Empty directory")))
                return
            
            # Add items to list
            for item in items:
                is_supported = item.is_dir() or item.suffix.lower() in self.supported_extensions
                file_item = SelectableFileItem(item, is_supported)
                file_list.append(file_item)
        
        except PermissionError:
            file_list.append(ListItem(Static("🔒 Permission denied")))
        except Exception as e:
            file_list.append(ListItem(Static(f"❌ Error: {str(e)}")))
    
    @on(Button.Pressed, "#up-btn")
    def go_up(self) -> None:
        """Navigate to parent directory."""
        if self.current_path.parent != self.current_path:
            self.current_path = self.current_path.parent
    
    @on(Button.Pressed, "#home-btn")
    def go_home(self) -> None:
        """Navigate to home directory."""
        self.current_path = Path.home()
    
    @on(Button.Pressed, "#cwd-btn")
    def go_cwd(self) -> None:
        """Navigate to current working directory."""
        self.current_path = Path.cwd()
    
    @on(Button.Pressed, "#select-all-btn")
    def select_all_supported(self) -> None:
        """Select all supported files in current directory."""
        file_list = self.query_one("#file-list", ListView)
        
        for item in file_list.children:
            if isinstance(item, SelectableFileItem):
                if item.is_supported and item.file_path.is_file():
                    item.set_selected(True)
                    self.selected_files.add(item.file_path)
        
        self.update_selection_display()
        self.post_message(self.FileSelected(list(self.selected_files)))
    
    @on(Button.Pressed, "#clear-selection-btn")
    def clear_selection(self) -> None:
        """Clear all selections."""
        file_list = self.query_one("#file-list", ListView)
        
        for item in file_list.children:
            if isinstance(item, SelectableFileItem):
                item.set_selected(False)
        
        self.selected_files.clear()
        self.update_selection_display()
        self.post_message(self.FileSelected(list(self.selected_files)))
    
    @on(ListView.Selected)
    def handle_item_selected(self, event: ListView.Selected) -> None:
        """Handle file/folder selection."""
        if isinstance(event.item, SelectableFileItem):
            file_item = event.item
            
            if file_item.file_path.is_dir():
                # Navigate into directory
                self.current_path = file_item.file_path
            else:
                # Toggle file selection
                if file_item.is_supported:
                    file_item.toggle_selection()
                    
                    if file_item.is_selected:
                        self.selected_files.add(file_item.file_path)
                    else:
                        self.selected_files.discard(file_item.file_path)
                    
                    self.update_selection_display()
                    self.post_message(self.FileSelected(list(self.selected_files)))
    
    def update_selection_display(self) -> None:
        """Update the selection count display."""
        count = len(self.selected_files)
        text = f"Selected: {count} file{'s' if count != 1 else ''}"
        self.query_one("#selection-count", Static).update(text)
    
    def get_selected_files(self) -> List[Path]:
        """Get list of selected files."""
        return list(self.selected_files)
    
    def add_selected_folder(self, folder_path: Path, recursive: bool = True) -> None:
        """Add all supported files from a folder to selection."""
        if not folder_path.is_dir():
            return
        
        pattern = "**/*" if recursive else "*"
        for ext in self.supported_extensions:
            files = list(folder_path.glob(f"{pattern}{ext}"))
            self.selected_files.update(files)
        
        self.update_selection_display()
        self.post_message(self.FileSelected(list(self.selected_files)))


class ArxivDownloader:
    """Handles arXiv paper download and PDF extraction."""
    
    @staticmethod
    async def search_papers(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv for papers matching the query."""
        if not ARXIV_AVAILABLE:
            raise ImportError("feedparser is required for arXiv integration. Install with: pip install feedparser")
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for arXiv integration. Install with: pip install aiohttp")
        
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        papers = []
                        for entry in feed.entries:
                            # Extract PDF URL
                            pdf_url = None
                            for link in entry.links:
                                if link.type == "application/pdf":
                                    pdf_url = link.href
                                    break
                            
                            if pdf_url:
                                papers.append({
                                    "title": entry.title,
                                    "authors": [author.name for author in entry.authors],
                                    "summary": entry.summary,
                                    "published": entry.published,
                                    "pdf_url": pdf_url,
                                    "arxiv_id": entry.id.split("/")[-1]
                                })
                        
                        return papers
                    else:
                        raise Exception(f"Failed to search arXiv: HTTP {response.status}")
        except Exception as e:
            log.error(f"Error searching arXiv: {e}")
            raise
    
    @staticmethod
    async def download_pdf(pdf_url: str, filename: str, download_dir: Path) -> Path:
        """Download a PDF from arXiv."""
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for PDF downloads. Install with: pip install aiohttp")
        
        download_path = download_dir / filename
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(download_path, 'wb') as f:
                            f.write(content)
                        return download_path
                    else:
                        raise Exception(f"Failed to download PDF: HTTP {response.status}")
        except Exception as e:
            log.error(f"Error downloading PDF: {e}")
            raise


class DocumentInserterModal(ModalScreen[Dict[str, Any]]):
    """Enhanced modal for document insertion with interactive file selection."""
    
    DEFAULT_CSS = """
    DocumentInserterModal {
        align: center middle;
    }

    DocumentInserterModal > Container {
        width: 95;
        height: 45;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    DocumentInserterModal .section {
        margin: 1 0;
        padding: 1;
        border: solid $secondary;
    }

    DocumentInserterModal .section-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    DocumentInserterModal Input, DocumentInserterModal Select {
        width: 100%;
        margin: 0 0 1 0;
    }

    DocumentInserterModal Button {
        margin: 0 1 0 0;
    }

    DocumentInserterModal ListView {
        height: 6;
        margin: 1 0;
    }

    DocumentInserterModal ProgressBar {
        margin: 1 0;
    }

    DocumentInserterModal .hint {
        color: $text-muted;
        text-style: italic;
    }

    DocumentInserterModal TextArea {
        height: 4;
        margin: 1 0;
    }

    # File explorer styles
    InteractiveFileExplorer {
        height: 18;
        border: solid $secondary;
    }

    InteractiveFileExplorer #nav-bar {
        height: 3;
        padding: 1;
        background: $primary 10%;
    }

    InteractiveFileExplorer #current-path {
        width: 1fr;
        padding: 0 1;
        content-align: left middle;
        text-style: italic;
    }

    InteractiveFileExplorer #file-list {
        height: 1fr;
    }

    InteractiveFileExplorer #selection-info {
        height: 3;
        padding: 1;
        background: $surface;
    }

    InteractiveFileExplorer #selection-count {
        width: 1fr;
        content-align: left middle;
        text-style: bold;
    }

    # File item styles
    SelectableFileItem {
        height: 1;
        background: transparent;
    }

    SelectableFileItem:hover {
        background: $primary 20%;
    }

    SelectableFileItem.selected {
        background: $accent 40%;
        color: $text;
        text-style: bold;
    }

    SelectableFileItem.unsupported {
        color: $text-muted;
        text-style: dim;
    }

    # Progress styles
    #progress-section {
        height: 12;
    }

    #log-area {
        height: 6;
        border: solid $secondary;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+a", "select_all", "Select All Files"),
        Binding("ctrl+c", "clear_selection", "Clear Selection"),
        Binding("ctrl+s", "search_arxiv", "Enhanced arXiv Search"),
        Binding("ctrl+shift+s", "basic_arxiv", "Basic arXiv Search"),
        Binding("enter", "insert_selected", "Insert Selected"),
    ]

    def __init__(self, default_path: Optional[Path] = None):
        super().__init__()
        self.default_path = default_path or Path.cwd()
        self.selected_files: List[Path] = []
        self.mode = "files"  # files, arxiv
        self.arxiv_papers: List[Dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("📂 Interactive Document Insertion", classes="section-title")
            
            # Mode selection
            with Container(classes="section"):
                yield Static("📋 Insertion Mode:", classes="section-title")
                with Horizontal():
                    options = [("📁 Browse Files & Folders", "files")]
                    
                    if ENHANCED_ARXIV_AVAILABLE:
                        options.append(("🚀 Enhanced arXiv Research", "enhanced_arxiv"))
                    elif ARXIV_AVAILABLE:
                        options.append(("🔬 Basic arXiv Search", "arxiv"))
                    
                    yield Select(options, value="files", id="mode-select")
                    yield Static("Use mouse/keyboard to navigate and select files", classes="hint")
            
            # File explorer section
            with Container(classes="section", id="file-section"):
                yield Static("🗂️ File Explorer & Selection:", classes="section-title")
                yield InteractiveFileExplorer(self.default_path, id="file-explorer")
                with Horizontal():
                    yield Button("➕ Add Current Folder", id="add-folder-btn", variant="success")
                    yield Checkbox("📁 Include subfolders", value=True, id="recursive-search")
                    yield Checkbox("⚡ Skip duplicates", value=True, id="skip-duplicates")
            
            # arXiv search section
            with Container(classes="section", id="arxiv-section"):
                yield Static("🔬 arXiv Paper Search:", classes="section-title")
                with Horizontal():
                    yield Input(placeholder="Search query (e.g., 'machine learning transformers')", id="arxiv-query")
                    yield Button("🔍 Search", id="arxiv-search-btn", variant="primary")
                yield ListView(id="arxiv-results")
                yield Static("", id="arxiv-status")
            
            # Selection summary
            with Container(classes="section"):
                yield Static("📊 Selection Summary:", classes="section-title")
                yield Static("No files selected", id="selection-summary")
                with Horizontal():
                    yield Button("🗑️ Clear All", id="clear-all-btn")
                    yield Button("📋 Show Details", id="show-details-btn")
            
            # Progress and status
            with Container(classes="section", id="progress-section"):
                yield Static("⚙️ Processing Progress:", classes="section-title")
                yield ProgressBar(id="progress-bar")
                yield Static("", id="status-text")
                yield TextArea("", read_only=True, id="log-area")
            
            # Action buttons
            with Horizontal():
                yield Button("🚀 Insert Selected", variant="primary", id="insert-btn", disabled=True)
                yield Button("❌ Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        """Initialize the modal."""
        self.update_selection_summary()
        
        # Initially hide arXiv and progress sections
        arxiv_section = self.query_one("#arxiv-section")
        progress_section = self.query_one("#progress-section")
        arxiv_section.display = False
        progress_section.display = False

    @on(Select.Changed, "#mode-select")
    def mode_changed(self, event: Select.Changed) -> None:
        """Handle mode selection change."""
        self.mode = event.value
        file_section = self.query_one("#file-section")
        arxiv_section = self.query_one("#arxiv-section")
        
        if self.mode == "enhanced_arxiv":
            # Launch enhanced arXiv modal
            self.launch_enhanced_arxiv()
            # Reset mode to files after launching modal
            mode_select = self.query_one("#mode-select", Select)
            mode_select.value = "files"
            self.mode = "files"
        elif self.mode == "arxiv":
            file_section.display = False
            arxiv_section.display = True
        else:
            file_section.display = True
            arxiv_section.display = False
        
        # Clear current selections when switching modes
        self.selected_files.clear()
        self.update_selection_summary()

    def launch_enhanced_arxiv(self) -> None:
        """Launch the enhanced arXiv research modal."""
        if not ENHANCED_ARXIV_AVAILABLE:
            self.app.notify("Enhanced arXiv features require additional packages", severity="error")
            return
        
        def handle_arxiv_result(result: dict | None) -> None:
            if result:
                # Enhanced arXiv modal returns its own result format
                self.dismiss(result)
        
        self.app.push_screen(EnhancedArxivSearchModal(), handle_arxiv_result)

    @on(InteractiveFileExplorer.FileSelected)
    def files_selected(self, event: InteractiveFileExplorer.FileSelected) -> None:
        """Handle file selection from the explorer."""
        self.selected_files = event.selected_files
        self.update_selection_summary()

    @on(Button.Pressed, "#add-folder-btn")
    def add_current_folder(self) -> None:
        """Add all files from current folder to selection."""
        file_explorer = self.query_one("#file-explorer", InteractiveFileExplorer)
        recursive = self.query_one("#recursive-search", Checkbox).value
        
        current_path = file_explorer.current_path
        if current_path.is_dir():
            file_explorer.add_selected_folder(current_path, recursive)

    def update_selection_summary(self) -> None:
        """Update the selection summary display."""
        count = len(self.selected_files)
        summary = self.query_one("#selection-summary", Static)
        insert_btn = self.query_one("#insert-btn", Button)
        
        if count == 0:
            summary.update("No files selected")
            insert_btn.disabled = True
        else:
            # Calculate total size
            total_size = 0
            file_types = set()
            
            for file_path in self.selected_files:
                try:
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_types.add(file_path.suffix.lower())
                except Exception:
                    pass
            
            # Format size
            if total_size < 1024:
                size_text = f"{total_size} B"
            elif total_size < 1024 * 1024:
                size_text = f"{total_size // 1024} KB"
            else:
                size_text = f"{total_size // (1024 * 1024)} MB"
            
            # Format types
            types_text = ", ".join(sorted(file_types)) if file_types else "various"
            
            summary.update(f"📊 {count} files selected ({size_text}) • Types: {types_text}")
            insert_btn.disabled = False

    @on(Button.Pressed, "#clear-all-btn")
    def clear_all_selections(self) -> None:
        """Clear all file selections."""
        file_explorer = self.query_one("#file-explorer", InteractiveFileExplorer)
        file_explorer.clear_selection()
        self.selected_files.clear()
        self.update_selection_summary()

    @on(Button.Pressed, "#show-details-btn")
    def show_selection_details(self) -> None:
        """Show detailed list of selected files."""
        if not self.selected_files:
            self.app.notify("No files selected", severity="warning")
            return
        
        details = []
        for i, file_path in enumerate(self.selected_files[:10], 1):  # Show first 10
            details.append(f"{i}. {file_path.name} ({file_path.parent})")
        
        if len(self.selected_files) > 10:
            details.append(f"... and {len(self.selected_files) - 10} more files")
        
        detail_text = "\n".join(details)
        self.app.notify(f"Selected Files:\n{detail_text}", timeout=10)

    @on(Button.Pressed, "#arxiv-search-btn")
    async def search_arxiv(self) -> None:
        """Search arXiv for papers."""
        if not ARXIV_AVAILABLE:
            self.app.notify("feedparser is required for arXiv integration", severity="error")
            return
        
        query_input = self.query_one("#arxiv-query", Input)
        query = query_input.value.strip()
        
        if not query:
            self.app.notify("Please enter a search query", severity="warning")
            return
        
        status = self.query_one("#arxiv-status", Static)
        results_list = self.query_one("#arxiv-results", ListView)
        
        status.update("🔍 Searching arXiv...")
        results_list.clear()
        
        try:
            papers = await ArxivDownloader.search_papers(query, max_results=10)
            self.arxiv_papers = papers
            
            if papers:
                for i, paper in enumerate(papers):
                    authors_str = ", ".join(paper["authors"][:3])
                    if len(paper["authors"]) > 3:
                        authors_str += " et al."
                    
                    item_content = Static(f"📄 {paper['title'][:80]}...\n👥 {authors_str}\n🔗 ID: {paper['arxiv_id']}")
                    selectable_item = SelectableFileItem(Path(f"arxiv_{paper['arxiv_id']}.pdf"), True)
                    selectable_item.query_one(Static).update(str(item_content))
                    results_list.append(selectable_item)
                
                status.update(f"✅ Found {len(papers)} papers")
            else:
                status.update("❌ No papers found")
        
        except Exception as e:
            status.update(f"❌ Error searching arXiv: {str(e)}")
            log.error(f"arXiv search error: {e}")

    @on(Button.Pressed, "#insert-btn")
    async def insert_documents(self) -> None:
        """Start the document insertion process."""
        try:
            if self.mode == "files" and not self.selected_files:
                self.app.notify("No files selected", severity="warning")
                return
            elif self.mode == "arxiv" and not self.arxiv_papers:
                self.app.notify("No arXiv papers found", severity="warning") 
                return
            
            # Show progress section
            progress_section = self.query_one("#progress-section")
            progress_section.display = True
            
            progress_bar = self.query_one("#progress-bar", ProgressBar)
            status_text = self.query_one("#status-text", Static)
            log_area = self.query_one("#log-area", TextArea)
            
            # Gather options
            skip_duplicates = self.query_one("#skip-duplicates", Checkbox).value
            recursive = self.query_one("#recursive-search", Checkbox).value if self.mode == "files" else True
            
            # Prepare files to process
            files_to_process = []
            download_dir = None
            
            if self.mode == "files":
                files_to_process = self.selected_files.copy()
                status_text.update(f"📂 Preparing to insert {len(files_to_process)} files...")
            
            elif self.mode == "arxiv":
                # Get selected arXiv papers from ListView
                results_list = self.query_one("#arxiv-results", ListView)
                selected_papers = []
                
                for item in results_list.children:
                    if isinstance(item, SelectableFileItem) and item.is_selected:
                        # Find corresponding paper
                        paper_id = item.file_path.stem.replace("arxiv_", "").replace("_", "/")
                        for paper in self.arxiv_papers:
                            if paper["arxiv_id"] == paper_id:
                                selected_papers.append(paper)
                                break
                
                if not selected_papers:
                    # If none explicitly selected, use all papers
                    selected_papers = self.arxiv_papers
                
                status_text.update(f"📥 Downloading {len(selected_papers)} arXiv papers...")
                
                # Download papers to temporary directory
                download_dir = Path(tempfile.mkdtemp(prefix="elia_arxiv_"))
                
                for i, paper in enumerate(selected_papers):
                    try:
                        filename = f"{paper['arxiv_id'].replace('/', '_')}.pdf"
                        downloaded_path = await ArxivDownloader.download_pdf(
                            paper['pdf_url'], filename, download_dir
                        )
                        files_to_process.append(downloaded_path)
                        progress_bar.progress = (i + 1) / len(selected_papers) * 50  # First 50% for downloads
                        log_area.insert(f"✅ Downloaded: {paper['title'][:50]}...\n")
                    except Exception as e:
                        log_area.insert(f"❌ Failed to download {paper['title'][:50]}...: {str(e)}\n")
            
            if not files_to_process:
                self.app.notify("No files to process", severity="warning")
                return
            
            # Prepare result data
            result = {
                "mode": "interactive_" + self.mode,
                "files": files_to_process,
                "options": {
                    "skip_duplicates": skip_duplicates,
                    "recursive": recursive,
                    "file_type_filter": "all"
                },
                "download_dir": download_dir,
                "total_files": len(files_to_process)
            }
            
            status_text.update(f"✅ Ready to insert {len(files_to_process)} files")
            self.dismiss(result)
            
        except Exception as e:
            log.error(f"Error preparing insertion: {e}")
            self.app.notify(f"Error: {str(e)}", severity="error")

    @on(Button.Pressed, "#cancel-btn")
    def cancel_insertion(self) -> None:
        """Cancel the insertion."""
        self.dismiss(None)

    # Action handlers for key bindings
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss(None)

    def action_select_all(self) -> None:
        """Select all files in current directory."""
        if self.mode == "files":
            file_explorer = self.query_one("#file-explorer", InteractiveFileExplorer)
            file_explorer.select_all_supported()

    def action_clear_selection(self) -> None:
        """Clear all selections."""
        if self.mode == "files":
            self.clear_all_selections()

    def action_search_arxiv(self) -> None:
        """Launch enhanced arXiv search."""
        if ENHANCED_ARXIV_AVAILABLE:
            self.launch_enhanced_arxiv()
        elif ARXIV_AVAILABLE:
            mode_select = self.query_one("#mode-select", Select)
            mode_select.value = "arxiv"
            self.mode_changed(Select.Changed(mode_select, "arxiv"))
            try:
                self.query_one("#arxiv-query", Input).focus()
            except Exception:
                pass
        else:
            self.app.notify("arXiv integration requires feedparser package", severity="error")

    def action_basic_arxiv(self) -> None:
        """Switch to basic arXiv mode."""
        if ARXIV_AVAILABLE:
            mode_select = self.query_one("#mode-select", Select)
            mode_select.value = "arxiv"
            self.mode_changed(Select.Changed(mode_select, "arxiv"))
            try:
                self.query_one("#arxiv-query", Input).focus()
            except Exception:
                pass
        else:
            self.app.notify("arXiv integration requires feedparser package", severity="error")

    def action_insert_selected(self) -> None:
        """Insert selected files."""
        if self.selected_files or (self.mode == "arxiv" and self.arxiv_papers):
            self.run_worker(self.insert_documents(), exclusive=True)
        else:
            self.app.notify("No files selected for insertion", severity="warning")