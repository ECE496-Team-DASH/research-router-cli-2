"""
Simple Document Insertion Widget - Claude Code Style

A clean, straightforward interface for inserting documents into nano-graphrag sessions.
Focuses on simplicity and ease of use.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

from textual import on, log
from textual.app import ComposeResult
from textual.widgets import (
    Static, Input, Button, ListView, ListItem, 
    Checkbox, DirectoryTree, Label
)
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.binding import Binding


class SimpleFileItem(ListItem):
    """Simple file item for display."""
    
    def __init__(self, file_path: Path, *args, **kwargs):
        self.file_path = file_path
        
        # Simple display format
        if file_path.is_dir():
            icon = "📁"
            name = f"{file_path.name}/"
        else:
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                icon = "📄"
            elif ext in [".txt", ".md"]:
                icon = "📝"
            elif ext == ".docx":
                icon = "📋"
            else:
                icon = "📄"
            name = file_path.name
        
        display_text = f"{icon} {name}"
        super().__init__(Static(display_text), *args, **kwargs)


class SimpleDocumentInserter(ModalScreen[Dict[str, Any]]):
    """Simple, Claude Code-style document insertion modal."""
    
    DEFAULT_CSS = """
    SimpleDocumentInserter {
        align: center middle;
    }

    SimpleDocumentInserter > Container {
        width: 80;
        height: 35;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }

    SimpleDocumentInserter .header {
        text-align: center;
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    SimpleDocumentInserter .section {
        margin: 1 0;
        padding: 1;
        border: solid $secondary;
    }

    SimpleDocumentInserter .section-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    SimpleDocumentInserter Input {
        width: 100%;
        margin: 0 0 1 0;
    }

    SimpleDocumentInserter ListView {
        height: 10;
        border: solid $secondary;
        margin: 1 0;
    }

    SimpleDocumentInserter .selected-files {
        height: 4;
        border: solid $secondary;
        margin: 1 0;
        padding: 1;
        background: $primary 10%;
    }

    SimpleDocumentInserter Button {
        margin: 0 1 0 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+a", "select_all", "Select All"),
        Binding("enter", "insert", "Insert Selected"),
    ]

    def __init__(self, default_path: Optional[Path] = None):
        super().__init__()
        self.current_path = default_path or Path.cwd()
        self.selected_files: List[Path] = []

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("📂 Insert Documents", classes="header")
            
            # Current location
            with Container(classes="section"):
                yield Static("📍 Current Location", classes="section-title")
                with Horizontal():
                    yield Static(str(self.current_path), id="current-path")
                    yield Button("🏠", id="home-btn", tooltip="Go to home directory")
                    yield Button("⬆️", id="up-btn", tooltip="Go up one level")
            
            # Quick path input
            with Container(classes="section"):
                yield Static("🚀 Quick Navigation", classes="section-title")
                yield Input(
                    placeholder="Enter path (e.g., /path/to/documents)", 
                    id="path-input"
                )
                yield Button("📁 Browse", id="browse-btn", variant="primary")
            
            # File list
            with Container(classes="section"):
                yield Static("📂 Files & Folders", classes="section-title")
                yield ListView(id="file-list")
                with Horizontal():
                    yield Button("📄 Select File", id="select-file-btn")
                    yield Button("📁 Select Folder", id="select-folder-btn")
                    yield Button("🗑️ Clear", id="clear-btn")
            
            # Selected files
            with Container(classes="selected-files"):
                yield Static("✅ Selected Files", classes="section-title")
                yield Static("No files selected", id="selected-display")
            
            # Options
            with Container(classes="section"):
                yield Static("⚙️ Options", classes="section-title")
                with Horizontal():
                    yield Checkbox("Include subfolders", value=True, id="recursive")
                    yield Checkbox("Skip duplicates", value=True, id="skip-duplicates")
            
            # Actions
            with Horizontal():
                yield Button("🚀 Insert Documents", variant="primary", id="insert-btn", disabled=True)
                yield Button("❌ Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        """Initialize the modal."""
        self.refresh_file_list()

    def refresh_file_list(self) -> None:
        """Refresh the file list for current directory."""
        file_list = self.query_one("#file-list", ListView)
        file_list.clear()
        
        try:
            if not self.current_path.exists() or not self.current_path.is_dir():
                file_list.append(ListItem(Static("❌ Invalid directory")))
                return
            
            items = sorted(self.current_path.iterdir(), 
                         key=lambda x: (x.is_file(), x.name.lower()))
            
            if not items:
                file_list.append(ListItem(Static("📂 Empty directory")))
                return
            
            for item in items:
                if not item.name.startswith('.'):  # Hide hidden files
                    file_list.append(SimpleFileItem(item))
        
        except PermissionError:
            file_list.append(ListItem(Static("🔒 Permission denied")))
        except Exception as e:
            file_list.append(ListItem(Static(f"❌ Error: {str(e)}")))

    @on(Button.Pressed, "#home-btn")
    def go_home(self) -> None:
        """Navigate to home directory."""
        self.current_path = Path.home()
        self.query_one("#current-path", Static).update(str(self.current_path))
        self.refresh_file_list()

    @on(Button.Pressed, "#up-btn")
    def go_up(self) -> None:
        """Navigate to parent directory."""
        if self.current_path.parent != self.current_path:
            self.current_path = self.current_path.parent
            self.query_one("#current-path", Static).update(str(self.current_path))
            self.refresh_file_list()

    @on(Button.Pressed, "#browse-btn")
    def browse_path(self) -> None:
        """Navigate to entered path."""
        path_input = self.query_one("#path-input", Input)
        path_str = path_input.value.strip()
        
        if path_str:
            try:
                new_path = Path(path_str).expanduser().resolve()
                if new_path.exists() and new_path.is_dir():
                    self.current_path = new_path
                    self.query_one("#current-path", Static).update(str(self.current_path))
                    self.refresh_file_list()
                    path_input.value = ""
                else:
                    self.app.notify("Path not found or not a directory", severity="error")
            except Exception as e:
                self.app.notify(f"Invalid path: {str(e)}", severity="error")

    @on(ListView.Selected)
    def handle_item_selected(self, event: ListView.Selected) -> None:
        """Handle file/folder selection."""
        if isinstance(event.item, SimpleFileItem):
            file_item = event.item
            
            if file_item.file_path.is_dir():
                # Navigate into directory
                self.current_path = file_item.file_path
                self.query_one("#current-path", Static).update(str(self.current_path))
                self.refresh_file_list()
            else:
                # Select file
                if file_item.file_path not in self.selected_files:
                    self.selected_files.append(file_item.file_path)
                    self.update_selected_display()

    @on(Button.Pressed, "#select-file-btn")
    def select_current_file(self) -> None:
        """Select currently highlighted file."""
        file_list = self.query_one("#file-list", ListView)
        if file_list.highlighted_child and isinstance(file_list.highlighted_child, SimpleFileItem):
            file_item = file_list.highlighted_child
            if file_item.file_path.is_file() and file_item.file_path not in self.selected_files:
                self.selected_files.append(file_item.file_path)
                self.update_selected_display()

    @on(Button.Pressed, "#select-folder-btn")
    def select_current_folder(self) -> None:
        """Select all files in current or highlighted folder."""
        file_list = self.query_one("#file-list", ListView)
        recursive = self.query_one("#recursive", Checkbox).value
        
        folder_path = self.current_path
        if file_list.highlighted_child and isinstance(file_list.highlighted_child, SimpleFileItem):
            if file_list.highlighted_child.file_path.is_dir():
                folder_path = file_list.highlighted_child.file_path
        
        # Add all supported files from folder
        supported_extensions = {'.txt', '.md', '.pdf', '.docx'}
        pattern = "**/*" if recursive else "*"
        
        for ext in supported_extensions:
            files = list(folder_path.glob(f"{pattern}{ext}"))
            for file_path in files:
                if file_path not in self.selected_files:
                    self.selected_files.append(file_path)
        
        self.update_selected_display()
        self.app.notify(f"Added files from {folder_path.name}", severity="success")

    @on(Button.Pressed, "#clear-btn")
    def clear_selection(self) -> None:
        """Clear all selected files."""
        self.selected_files.clear()
        self.update_selected_display()

    def update_selected_display(self) -> None:
        """Update the selected files display."""
        display = self.query_one("#selected-display", Static)
        insert_btn = self.query_one("#insert-btn", Button)
        
        if not self.selected_files:
            display.update("No files selected")
            insert_btn.disabled = True
        else:
            count = len(self.selected_files)
            if count <= 3:
                file_names = [f.name for f in self.selected_files]
                display.update(f"{count} files: {', '.join(file_names)}")
            else:
                first_three = [f.name for f in self.selected_files[:3]]
                display.update(f"{count} files: {', '.join(first_three)}... (+{count-3} more)")
            insert_btn.disabled = False

    @on(Button.Pressed, "#insert-btn")
    def insert_documents(self) -> None:
        """Insert selected documents."""
        if not self.selected_files:
            self.app.notify("No files selected", severity="warning")
            return
        
        recursive = self.query_one("#recursive", Checkbox).value
        skip_duplicates = self.query_one("#skip-duplicates", Checkbox).value
        
        result = {
            "mode": "simple_files",
            "files": self.selected_files,
            "options": {
                "skip_duplicates": skip_duplicates,
                "recursive": recursive,
                "file_type_filter": "all"
            },
            "total_files": len(self.selected_files)
        }
        
        self.dismiss(result)

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
        supported_extensions = {'.txt', '.md', '.pdf', '.docx'}
        
        for item_path in self.current_path.iterdir():
            if (item_path.is_file() and 
                item_path.suffix.lower() in supported_extensions and
                item_path not in self.selected_files):
                self.selected_files.append(item_path)
        
        self.update_selected_display()

    def action_insert(self) -> None:
        """Insert selected files."""
        if self.selected_files:
            self.insert_documents()
        else:
            self.app.notify("No files selected", severity="warning")