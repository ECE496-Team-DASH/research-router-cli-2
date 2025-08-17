"""
Paper Collection Management System

Organize, tag, and manage research paper collections for enhanced productivity
and knowledge organization in nano-graphrag sessions.
"""

import json
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict

from textual import on, log
from textual.app import ComposeResult
from textual.widgets import (
    Static, Input, Button, ListView, ListItem, Tree, TreeNode,
    Select, Label, TextArea, Tabs, TabPane
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.message import Message

try:
    from .enhanced_arxiv import ArxivPaper
    ARXIV_INTEGRATION = True
except ImportError:
    ARXIV_INTEGRATION = False


@dataclass
class PaperCollection:
    """A collection of research papers with metadata."""
    id: str
    name: str
    description: str
    papers: List[str] = field(default_factory=list)  # arXiv IDs
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    color: str = "blue"
    is_smart: bool = False  # Smart collections auto-update based on criteria
    smart_criteria: Dict[str, Any] = field(default_factory=dict)
    
    def add_paper(self, arxiv_id: str):
        """Add a paper to the collection."""
        if arxiv_id not in self.papers:
            self.papers.append(arxiv_id)
            self.updated_at = datetime.now()
    
    def remove_paper(self, arxiv_id: str):
        """Remove a paper from the collection."""
        if arxiv_id in self.papers:
            self.papers.remove(arxiv_id)
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['tags'] = list(self.tags)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperCollection':
        """Create from dictionary."""
        data['tags'] = set(data.get('tags', []))
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class CollectionManager:
    """Manages paper collections with persistence."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.home() / ".elia_cache" / "collections"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.collections_file = self.storage_dir / "collections.json"
        self.collections: Dict[str, PaperCollection] = {}
        self.tags: Set[str] = set()
        self.load_collections()
    
    def load_collections(self):
        """Load collections from disk."""
        try:
            if self.collections_file.exists():
                with open(self.collections_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for collection_data in data.get('collections', []):
                        collection = PaperCollection.from_dict(collection_data)
                        self.collections[collection.id] = collection
                        self.tags.update(collection.tags)
        
        except Exception as e:
            log.error(f"Failed to load collections: {e}")
    
    def save_collections(self):
        """Save collections to disk."""
        try:
            data = {
                'collections': [collection.to_dict() for collection in self.collections.values()],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.collections_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        
        except Exception as e:
            log.error(f"Failed to save collections: {e}")
    
    def create_collection(self, name: str, description: str = "", color: str = "blue") -> PaperCollection:
        """Create a new collection."""
        collection_id = f"collection_{len(self.collections) + 1}_{int(datetime.now().timestamp())}"
        
        collection = PaperCollection(
            id=collection_id,
            name=name,
            description=description,
            color=color
        )
        
        self.collections[collection_id] = collection
        self.save_collections()
        return collection
    
    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection."""
        if collection_id in self.collections:
            del self.collections[collection_id]
            self.save_collections()
            return True
        return False
    
    def add_paper_to_collection(self, collection_id: str, arxiv_id: str) -> bool:
        """Add a paper to a collection."""
        if collection_id in self.collections:
            self.collections[collection_id].add_paper(arxiv_id)
            self.save_collections()
            return True
        return False
    
    def remove_paper_from_collection(self, collection_id: str, arxiv_id: str) -> bool:
        """Remove a paper from a collection."""
        if collection_id in self.collections:
            self.collections[collection_id].remove_paper(arxiv_id)
            self.save_collections()
            return True
        return False
    
    def get_collections_for_paper(self, arxiv_id: str) -> List[PaperCollection]:
        """Get all collections containing a specific paper."""
        return [collection for collection in self.collections.values() 
                if arxiv_id in collection.papers]
    
    def search_collections(self, query: str) -> List[PaperCollection]:
        """Search collections by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for collection in self.collections.values():
            if (query_lower in collection.name.lower() or 
                query_lower in collection.description.lower() or
                any(query_lower in tag.lower() for tag in collection.tags)):
                results.append(collection)
        
        return results
    
    def get_all_tags(self) -> Set[str]:
        """Get all tags used across collections."""
        all_tags = set()
        for collection in self.collections.values():
            all_tags.update(collection.tags)
        return all_tags


class CollectionTree(Tree):
    """Tree widget for displaying collections hierarchy."""
    
    class CollectionSelected(Message):
        """Message sent when collection is selected."""
        def __init__(self, collection: PaperCollection) -> None:
            self.collection = collection
            super().__init__()
    
    def __init__(self, collection_manager: CollectionManager, **kwargs):
        super().__init__("📚 Research Collections", **kwargs)
        self.collection_manager = collection_manager
        self.refresh_tree()
    
    def refresh_tree(self):
        """Refresh the collections tree."""
        self.clear()
        
        # Group collections by tags
        tag_groups: Dict[str, List[PaperCollection]] = {}
        untagged_collections = []
        
        for collection in self.collection_manager.collections.values():
            if collection.tags:
                for tag in collection.tags:
                    if tag not in tag_groups:
                        tag_groups[tag] = []
                    tag_groups[tag].append(collection)
            else:
                untagged_collections.append(collection)
        
        # Add tag groups
        for tag, collections in sorted(tag_groups.items()):
            tag_node = self.root.add(f"🏷️ {tag}")
            for collection in sorted(collections, key=lambda c: c.name):
                collection_node = tag_node.add(
                    f"📁 {collection.name} ({len(collection.papers)} papers)",
                    data=collection
                )
        
        # Add untagged collections
        if untagged_collections:
            untagged_node = self.root.add("📂 Untagged")
            for collection in sorted(untagged_collections, key=lambda c: c.name):
                untagged_node.add(
                    f"📁 {collection.name} ({len(collection.papers)} papers)",
                    data=collection
                )
        
        self.root.expand()
    
    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        if event.node.data and isinstance(event.node.data, PaperCollection):
            self.post_message(self.CollectionSelected(event.node.data))


class CollectionDetailsPanel(ScrollableContainer):
    """Panel showing detailed collection information."""
    
    DEFAULT_CSS = """
    CollectionDetailsPanel {
        border: solid $secondary;
        padding: 1;
    }
    
    CollectionDetailsPanel .collection-header {
        margin: 0 0 2 0;
        padding: 1;
        background: $primary 10%;
        border: solid $primary;
    }
    
    CollectionDetailsPanel .collection-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    
    CollectionDetailsPanel .collection-meta {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    CollectionDetailsPanel .paper-item {
        margin: 0 0 1 0;
        padding: 1;
        border: solid $secondary 50%;
    }
    
    CollectionDetailsPanel .paper-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    CollectionDetailsPanel .paper-authors {
        color: $text-muted;
        text-style: italic;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_collection: Optional[PaperCollection] = None
    
    def display_collection(self, collection: PaperCollection):
        """Display collection details."""
        self.current_collection = collection
        asyncio.create_task(self._refresh_display())
    
    async def _refresh_display(self):
        """Refresh the display with current collection."""
        await self.remove_children()
        
        if not self.current_collection:
            await self.mount(Static("Select a collection to view details"))
            return
        
        collection = self.current_collection
        
        # Collection header
        with self.mount(Container(classes="collection-header")):
            await self.mount(Static(f"📁 {collection.name}", classes="collection-title"))
            await self.mount(Static(collection.description or "No description", classes="collection-meta"))
            
            # Metadata
            created_date = collection.created_at.strftime("%Y-%m-%d")
            updated_date = collection.updated_at.strftime("%Y-%m-%d")
            tags_str = ", ".join(collection.tags) if collection.tags else "No tags"
            
            await self.mount(Static(f"📅 Created: {created_date} • Updated: {updated_date}", classes="collection-meta"))
            await self.mount(Static(f"🏷️ Tags: {tags_str}", classes="collection-meta"))
            await self.mount(Static(f"📊 Papers: {len(collection.papers)}", classes="collection-meta"))
        
        # Papers list
        if collection.papers:
            await self.mount(Static("📄 Papers in Collection:", classes="collection-title"))
            
            for i, paper_id in enumerate(collection.papers, 1):
                with self.mount(Container(classes="paper-item")):
                    await self.mount(Static(f"{i}. arXiv:{paper_id}", classes="paper-title"))
                    # Note: In a full implementation, we'd fetch paper details from cache
                    await self.mount(Static("Paper details would be shown here", classes="paper-authors"))
        else:
            await self.mount(Static("📂 This collection is empty"))


class CollectionManagementModal(ModalScreen[Optional[PaperCollection]]):
    """Modal for managing paper collections."""
    
    DEFAULT_CSS = """
    CollectionManagementModal {
        align: center middle;
    }

    CollectionManagementModal > Container {
        width: 90;
        height: 40;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }

    CollectionManagementModal Tabs {
        height: 1fr;
    }

    CollectionManagementModal .form-section {
        margin: 1 0;
        padding: 1;
        border: solid $secondary;
    }

    CollectionManagementModal .form-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    CollectionManagementModal Input, CollectionManagementModal TextArea {
        width: 100%;
        margin: 0 0 1 0;
    }

    CollectionManagementModal .action-buttons {
        height: 3;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+n", "new_collection", "New Collection"),
        Binding("ctrl+s", "save_collection", "Save Collection"),
    ]

    def __init__(self, collection_manager: CollectionManager, **kwargs):
        super().__init__(**kwargs)
        self.collection_manager = collection_manager
        self.selected_collection: Optional[PaperCollection] = None

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("📚 Collection Management", classes="form-title")

            with Tabs():
                # Browse collections tab
                with TabPane("📂 Browse", id="browse-tab"):
                    with Horizontal():
                        # Collections tree
                        with Vertical():
                            yield Static("Collections:", classes="form-title")
                            yield CollectionTree(self.collection_manager, id="collections-tree")
                            
                            # Quick actions
                            with Horizontal():
                                yield Button("➕ New", id="new-collection-btn", variant="primary")
                                yield Button("✏️ Edit", id="edit-collection-btn", disabled=True)
                                yield Button("🗑️ Delete", id="delete-collection-btn", disabled=True)
                        
                        # Collection details
                        with Vertical():
                            yield Static("Collection Details:", classes="form-title")
                            yield CollectionDetailsPanel(id="collection-details")

                # Create/Edit collection tab
                with TabPane("✏️ Edit", id="edit-tab"):
                    with Container(classes="form-section"):
                        yield Static("Collection Information:", classes="form-title")
                        yield Label("Name:")
                        yield Input(placeholder="Enter collection name", id="collection-name")
                        yield Label("Description:")
                        yield TextArea(placeholder="Enter collection description", id="collection-description")
                        yield Label("Tags (comma-separated):")
                        yield Input(placeholder="e.g., machine-learning, transformers, nlp", id="collection-tags")
                        yield Label("Color:")
                        yield Select([
                            ("Blue", "blue"),
                            ("Green", "green"),
                            ("Red", "red"),
                            ("Purple", "purple"),
                            ("Orange", "orange")
                        ], value="blue", id="collection-color")

            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("💾 Save Collection", variant="primary", id="save-btn")
                yield Button("❌ Cancel", variant="default", id="cancel-btn")

    @on(CollectionTree.CollectionSelected)
    def collection_selected(self, event: CollectionTree.CollectionSelected) -> None:
        """Handle collection selection."""
        self.selected_collection = event.collection
        
        # Update details panel
        details_panel = self.query_one("#collection-details", CollectionDetailsPanel)
        details_panel.display_collection(event.collection)
        
        # Enable edit/delete buttons
        self.query_one("#edit-collection-btn", Button).disabled = False
        self.query_one("#delete-collection-btn", Button).disabled = False

    @on(Button.Pressed, "#new-collection-btn")
    def new_collection(self) -> None:
        """Start creating new collection."""
        self.selected_collection = None
        self._clear_form()
        
        # Switch to edit tab
        tabs = self.query_one(Tabs)
        tabs.active = "edit-tab"

    @on(Button.Pressed, "#edit-collection-btn")
    def edit_collection(self) -> None:
        """Edit selected collection."""
        if not self.selected_collection:
            return
        
        self._populate_form(self.selected_collection)
        
        # Switch to edit tab
        tabs = self.query_one(Tabs)
        tabs.active = "edit-tab"

    @on(Button.Pressed, "#delete-collection-btn")
    def delete_collection(self) -> None:
        """Delete selected collection."""
        if not self.selected_collection:
            return
        
        # Confirm deletion
        self.app.notify(
            f"Collection '{self.selected_collection.name}' deleted", 
            severity="warning"
        )
        
        self.collection_manager.delete_collection(self.selected_collection.id)
        
        # Refresh tree
        tree = self.query_one("#collections-tree", CollectionTree)
        tree.refresh_tree()
        
        self.selected_collection = None

    @on(Button.Pressed, "#save-btn")
    def save_collection(self) -> None:
        """Save collection."""
        name = self.query_one("#collection-name", Input).value.strip()
        if not name:
            self.app.notify("Collection name is required", severity="error")
            return
        
        description = self.query_one("#collection-description", TextArea).text.strip()
        tags_str = self.query_one("#collection-tags", Input).value.strip()
        color = self.query_one("#collection-color", Select).value
        
        tags = set(tag.strip() for tag in tags_str.split(",") if tag.strip())
        
        if self.selected_collection:
            # Update existing collection
            self.selected_collection.name = name
            self.selected_collection.description = description
            self.selected_collection.tags = tags
            self.selected_collection.color = color
            self.selected_collection.updated_at = datetime.now()
            self.collection_manager.save_collections()
        else:
            # Create new collection
            collection = self.collection_manager.create_collection(name, description, color)
            collection.tags = tags
            self.collection_manager.save_collections()
            self.selected_collection = collection
        
        # Refresh tree and switch back to browse tab
        tree = self.query_one("#collections-tree", CollectionTree)
        tree.refresh_tree()
        
        tabs = self.query_one(Tabs)
        tabs.active = "browse-tab"
        
        self.app.notify(f"Collection '{name}' saved successfully", severity="success")

    @on(Button.Pressed, "#cancel-btn")
    def cancel_management(self) -> None:
        """Cancel collection management."""
        self.dismiss(None)

    def _clear_form(self):
        """Clear the form fields."""
        self.query_one("#collection-name", Input).value = ""
        self.query_one("#collection-description", TextArea).text = ""
        self.query_one("#collection-tags", Input).value = ""
        self.query_one("#collection-color", Select).value = "blue"

    def _populate_form(self, collection: PaperCollection):
        """Populate form with collection data."""
        self.query_one("#collection-name", Input).value = collection.name
        self.query_one("#collection-description", TextArea).text = collection.description
        self.query_one("#collection-tags", Input).value = ", ".join(collection.tags)
        self.query_one("#collection-color", Select).value = collection.color

    # Action handlers
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss(None)

    def action_new_collection(self) -> None:
        """Handle new collection shortcut."""
        self.new_collection()

    def action_save_collection(self) -> None:
        """Handle save collection shortcut."""
        self.save_collection()