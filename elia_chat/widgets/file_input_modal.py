"""
Simple File Input Modal for F9 Document Insertion

Just a simple input field to enter a file path - no fancy UI, no popups.
"""

from textual import on
from textual.app import ComposeResult
from textual.widgets import Static, Input, Button
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.binding import Binding


class FileInputModal(ModalScreen[str]):
    """Simple modal for entering a file path."""
    
    DEFAULT_CSS = """
    FileInputModal {
        align: center middle;
    }

    FileInputModal > Container {
        width: 60;
        height: 8;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }

    FileInputModal Input {
        width: 100%;
        margin: 1 0;
    }

    FileInputModal Button {
        margin: 0 1 0 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "submit", "Insert"),
    ]

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("📄 Insert Document")
            yield Static("Enter file or folder path:")
            yield Input(placeholder="e.g., /path/to/document.pdf or /path/to/folder", id="file-path")
            with Horizontal():
                yield Button("Insert", variant="primary", id="insert-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        """Focus the input when mounted."""
        self.query_one("#file-path", Input).focus()

    @on(Button.Pressed, "#insert-btn")
    def insert_file(self) -> None:
        """Insert the file."""
        file_path = self.query_one("#file-path", Input).value.strip()
        if file_path:
            self.dismiss(file_path)
        else:
            self.app.notify("Please enter a file path", severity="warning")

    @on(Button.Pressed, "#cancel-btn")
    def cancel_insertion(self) -> None:
        """Cancel the insertion."""
        self.dismiss(None)

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key in input."""
        file_path = event.value.strip()
        if file_path:
            self.dismiss(file_path)

    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss(None)

    def action_submit(self) -> None:
        """Handle enter key."""
        self.insert_file()