from textual import on, log
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen, ModalScreen
from textual.widgets import Footer, Input, Static
from textual.containers import Center, Middle, Container, Vertical
import os
import asyncio

from elia_chat.chats_manager import ChatsManager
from elia_chat.widgets.agent_is_typing import ResponseStatus
from elia_chat.widgets.chat import Chat
from elia_chat.models import ChatData


class ChatScreen(Screen[None]):
    AUTO_FOCUS = "ChatPromptInput"
    BINDINGS = [
        Binding(
            key="escape",
            action="app.focus('prompt')",
            description="Focus prompt",
            key_display="esc",
            tooltip="Return focus to the prompt input.",
        ),
        Binding(
            key="f9",
            action="insert_document",
            description="Insert docs",
            key_display="F9",
            tooltip="Insert documents into nano-graphrag session.",
        ),
    ]

    def __init__(
        self,
        chat_data: ChatData,
    ):
        super().__init__()
        self.chat_data = chat_data
        self.chats_manager = ChatsManager()

    def compose(self) -> ComposeResult:
        yield Chat(self.chat_data)
        yield Footer()

    @on(Chat.NewUserMessage)
    def new_user_message(self, event: Chat.NewUserMessage) -> None:
        """Handle a new user message."""
        self.query_one(Chat).allow_input_submit = False
        response_status = self.query_one(ResponseStatus)
        response_status.set_awaiting_response()
        response_status.display = True

    @on(Chat.AgentResponseStarted)
    def start_awaiting_response(self) -> None:
        """Prevent sending messages because the agent is typing."""
        response_status = self.query_one(ResponseStatus)
        response_status.set_agent_responding()
        response_status.display = True

    @on(Chat.AgentResponseComplete)
    async def agent_response_complete(self, event: Chat.AgentResponseComplete) -> None:
        """Allow the user to send messages again."""
        self.query_one(ResponseStatus).display = False
        self.query_one(Chat).allow_input_submit = True
        log.debug(
            f"Agent response complete. Adding message "
            f"to chat_id {event.chat_id!r}: {event.message}"
        )
        if self.chat_data.id is None:
            raise RuntimeError("Chat has no ID. This is likely a bug in Elia.")

        await self.chats_manager.add_message_to_chat(
            chat_id=self.chat_data.id, message=event.message
        )

    def action_insert_document(self) -> None:
        """Action to insert a document into nano-graphrag session."""
        # Check if this is a nano-graphrag model
        model_name = self.chat_data.model.name
        if model_name != "nano-graphrag":
            self.app.notify(
                "Document insertion is only available for Nano-GraphRAG models.",
                title="Feature Not Available",
                severity="warning",
            )
            return
        
        # Simple file path input - minimal modal
        from elia_chat.widgets.file_input_modal import FileInputModal
        
        def handle_file_input(file_path: str) -> None:
            if file_path and file_path.strip():
                self.run_worker(self._insert_document_into_nanographrag(file_path.strip()), exclusive=False)

        self.app.push_screen(FileInputModal(), handle_file_input)
    
    async def _process_enhanced_insertion(self, result: dict) -> None:
        """Process the enhanced document insertion with bulk operations and arXiv support."""
        import os, asyncio, shutil
        from datetime import datetime, timezone
        from pathlib import Path
        from elia_chat.models import ChatMessage
        from elia_chat.widgets.chatbox import Chatbox
        from elia_chat.widgets.agent_is_typing import ResponseStatus

        response_chatbox = None
        response_status = None
        temp_cleanup_dirs = []
        
        try:
            model_name = self.chat_data.model.name
            files_to_process = result["files"]
            mode = result["mode"]
            options = result["options"]
            download_dir = result.get("download_dir")
            
            if download_dir:
                temp_cleanup_dirs.append(download_dir)

            # Prepare streaming chatbox
            ai_message = {"content": "", "role": "assistant"}
            now = datetime.now(timezone.utc)
            cm = ChatMessage(message=ai_message, model=self.chat_data.model, timestamp=now)
            response_chatbox = Chatbox(message=cm, model=self.chat_data.model, classes="response-in-progress")
            
            if mode == "simple_files":
                if len(files_to_process) == 1:
                    response_chatbox.border_title = "Inserting document into GraphRAG..."
                else:
                    response_chatbox.border_title = f"Inserting {len(files_to_process)} documents into GraphRAG..."
            elif mode == "single":
                response_chatbox.border_title = "Inserting document into GraphRAG..."
            elif mode == "folder":
                response_chatbox.border_title = f"Inserting {len(files_to_process)} documents from folder..."
            elif mode == "arxiv":
                response_chatbox.border_title = f"Inserting {len(files_to_process)} arXiv papers..."

            chat_widget = self.query_one(Chat)
            chat_container = chat_widget.query_one("#chat-container")
            await chat_container.mount(response_chatbox)

            response_status = self.query_one(ResponseStatus)
            response_status.set_agent_responding()
            response_status.display = True

            # Load nanographrag model
            from elia_chat.nanographrag_model import get_nanographrag_model
            from elia_chat.config import load_nanographrag_config
            nanographrag_model = get_nanographrag_model(model_name, load_nanographrag_config())
            if not nanographrag_model:
                response_chatbox.append_chunk("❌ Failed to get nano-graphrag model instance\n")
                return
            if not self.chat_data.id:
                response_chatbox.append_chunk("❌ Chat session not initialized\n")
                return

            # Display overview
            response_chatbox.append_chunk(f"🚀 Starting {mode} insertion mode\n")
            response_chatbox.append_chunk(f"📊 Total files to process: {len(files_to_process)}\n")
            if options["skip_duplicates"]:
                response_chatbox.append_chunk("⚠️ Skipping duplicates: Enabled\n")
            response_chatbox.append_chunk("\n")

            # Process files
            successful = 0
            skipped = 0
            failed = 0
            
            for i, file_path in enumerate(files_to_process, 1):
                filename = file_path.name
                response_chatbox.append_chunk(f"📄 [{i}/{len(files_to_process)}] Processing: {filename}\n")
                
                collected_lines = []
                def push_line(line: str):
                    collected_lines.append(line)

                try:
                    # Check if file should be skipped due to duplicates
                    session = nanographrag_model.get_or_create_session(self.chat_data.id)
                    doc_key = str(file_path.absolute())
                    
                    if options["skip_duplicates"] and doc_key in session.indexed_documents:
                        response_chatbox.append_chunk(f"   ⏭️ Already indexed - skipping\n")
                        skipped += 1
                        continue
                    
                    # Insert document
                    result_data = await nanographrag_model.insert_document(
                        self.chat_data.id, str(file_path), line_callback=push_line
                    )
                    
                    if "error" in result_data:
                        response_chatbox.append_chunk(f"   ❌ Error: {result_data['error']}\n")
                        failed += 1
                    elif result_data.get("status") == "skipped":
                        response_chatbox.append_chunk(f"   ⚠️ Skipped: {result_data.get('message', 'Already indexed')}\n")
                        skipped += 1
                    else:
                        response_chatbox.append_chunk(f"   ✅ Success: {result_data.get('message', 'Indexed successfully')}\n")
                        successful += 1
                    
                    # Show brief logs if there were any significant issues
                    if collected_lines and any("error" in line.lower() or "warning" in line.lower() for line in collected_lines):
                        response_chatbox.append_chunk(f"   📋 Key logs: {len(collected_lines)} entries\n")
                        for line in collected_lines[-3:]:  # Show last 3 lines
                            if line.strip() and ("error" in line.lower() or "warning" in line.lower()):
                                response_chatbox.append_chunk(f"      {line.strip()}\n")
                    
                except Exception as e:
                    response_chatbox.append_chunk(f"   ❌ Exception: {str(e)}\n")
                    failed += 1
                
                # Add spacing between files
                response_chatbox.append_chunk("\n")
                
                # Brief pause for UI responsiveness
                await asyncio.sleep(0.1)

            # Summary
            response_chatbox.append_chunk("📊 **Insertion Summary:**\n")
            response_chatbox.append_chunk(f"   ✅ Successfully indexed: {successful}\n")
            response_chatbox.append_chunk(f"   ⏭️ Skipped (duplicates): {skipped}\n")
            response_chatbox.append_chunk(f"   ❌ Failed: {failed}\n")
            response_chatbox.append_chunk(f"   📁 Total processed: {len(files_to_process)}\n")

            if successful > 0:
                session_info = nanographrag_model.get_session_info(self.chat_data.id)
                if session_info:
                    response_chatbox.append_chunk(f"\n📚 Total documents in session: {session_info.get('indexed_documents', 0)}\n")
                    response_chatbox.append_chunk(f"📂 Session storage: {session_info.get('working_dir', 'Unknown')}\n")

            if mode == "arxiv":
                response_chatbox.append_chunk("\n🔗 arXiv papers are now available for querying!\n")
            
            response_chatbox.append_chunk("\n🎉 Bulk insertion completed! You can now ask questions about the content.\n")

            # Persist chat message
            self.chat_data.messages.append(cm)
            if self.chat_data.id:
                await self.chats_manager.add_message_to_chat(chat_id=self.chat_data.id, message=response_chatbox.message)

        except Exception as exc:
            if response_chatbox:
                response_chatbox.append_chunk(f"\n❌ Error during bulk insertion: {exc}\n")
                response_chatbox.border_title = "Bulk insertion failed"
        finally:
            # Cleanup temporary directories
            for temp_dir in temp_cleanup_dirs:
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                except Exception as e:
                    log.error(f"Failed to cleanup temp directory {temp_dir}: {e}")
            
            if response_chatbox:
                response_chatbox.border_title = response_chatbox.border_title or "Bulk insertion completed"
                response_chatbox.remove_class("response-in-progress")
            if response_status:
                response_status.display = False

    async def _insert_document_into_nanographrag(self, file_path: str) -> None:
        import os, asyncio
        from datetime import datetime, timezone
        from elia_chat.models import ChatMessage
        from elia_chat.widgets.chatbox import Chatbox
        from elia_chat.widgets.agent_is_typing import ResponseStatus

        response_chatbox = None
        response_status = None
        try:
            model_name = self.chat_data.model.name
            filename = os.path.basename(file_path)

            # Prepare streaming chatbox
            ai_message = {"content": "", "role": "assistant"}
            now = datetime.now(timezone.utc)
            cm = ChatMessage(message=ai_message, model=self.chat_data.model, timestamp=now)
            response_chatbox = Chatbox(message=cm, model=self.chat_data.model, classes="response-in-progress")
            response_chatbox.border_title = "Inserting document into GraphRAG..."

            chat_widget = self.query_one(Chat)
            chat_container = chat_widget.query_one("#chat-container")
            await chat_container.mount(response_chatbox)

            response_status = self.query_one(ResponseStatus)
            response_status.set_agent_responding()
            response_status.display = True

            # Collect lines instead of streaming immediately (user wants full output after completion)
            collected_lines: list[str] = []
            def push_line(line: str):
                collected_lines.append(line)

            # Intro lines
            response_chatbox.append_chunk(f"📄 Starting document insertion: {filename}\n")
            response_chatbox.append_chunk(f"📁 File path: {file_path}\n\n")

            from elia_chat.nanographrag_model import get_nanographrag_model
            from elia_chat.config import load_nanographrag_config
            nanographrag_model = get_nanographrag_model(model_name, load_nanographrag_config())
            if not nanographrag_model:
                response_chatbox.append_chunk("❌ Failed to get nano-graphrag model instance\n")
                return
            if not self.chat_data.id:
                response_chatbox.append_chunk("❌ Chat session not initialized\n")
                return

            response_chatbox.append_chunk("🔄 Initializing GraphRAG session...\n")
            await asyncio.sleep(0.05)
            response_chatbox.append_chunk("📖 Reading and processing document...\n")
            await asyncio.sleep(0.05)
            response_chatbox.append_chunk("▶️ Starting document insertion...\n")

            result = await nanographrag_model.insert_document(self.chat_data.id, file_path, line_callback=push_line)
            # After insertion, flush collected internal lines as a block
            if collected_lines:
                response_chatbox.append_chunk("\n--- Detailed Insert Log ---\n")
                # Collapse duplicate blank lines and format
                last_blank = False
                for raw in collected_lines:
                    line = raw.replace('\r', '')
                    if not line.strip():
                        if last_blank:
                            continue
                        last_blank = True
                    else:
                        last_blank = False
                    response_chatbox.append_chunk(line if line.endswith('\n') else line + '\n')
                response_chatbox.append_chunk("--- End Insert Log ---\n\n")
            response_chatbox.append_chunk("✅ Insertion finished (model returned)\n")

            if "error" in result:
                response_chatbox.append_chunk(f"\n❌ Error: {result['error']}\n")
            else:
                msg = result.get("message", "Document inserted successfully")
                if result.get("status") == "skipped":
                    response_chatbox.append_chunk(f"\n⚠️ Skipped: {msg}\n")
                else:
                    response_chatbox.append_chunk(f"\n✅ {msg}\n")
                    session_info = nanographrag_model.get_session_info(self.chat_data.id)
                    if session_info:
                        response_chatbox.append_chunk(f"📚 Total documents indexed in session: {session_info.get('indexed_documents', 0)}\n")
                        if 'working_dir' in session_info:
                            response_chatbox.append_chunk(f"📂 Session storage: {session_info['working_dir']}\n")

            response_chatbox.append_chunk("\n🎉 Document insertion completed! You can now ask questions about the content.\n")

            # Persist chat message
            self.chat_data.messages.append(cm)
            if self.chat_data.id:
                await self.chats_manager.add_message_to_chat(chat_id=self.chat_data.id, message=response_chatbox.message)

        except Exception as exc:
            if response_chatbox:
                response_chatbox.append_chunk(f"\n❌ Error inserting document: {exc}\n")
                response_chatbox.border_title = "Document insertion failed"
        finally:
            if response_chatbox:
                response_chatbox.border_title = response_chatbox.border_title or "Document insertion completed"
                response_chatbox.remove_class("response-in-progress")
            if response_status:
                response_status.display = False
    
    async def _add_system_message(self, message: str) -> None:
        """Add a system message to the chat."""
        try:
            from datetime import datetime, timezone
            from elia_chat.models import ChatMessage
            from elia_chat.widgets.chatbox import Chatbox
            
            # Create a system message
            system_message_data = {
                "role": "system",
                "content": message
            }
            
            now = datetime.now(timezone.utc)
            chat_message = ChatMessage(
                message=system_message_data,
                model=self.chat_data.model,
                timestamp=now
            )
            
            # Add to chat data
            self.chat_data.messages.append(chat_message)
            
            # Create a chatbox for the system message and add it to the chat container
            chat_widget = self.query_one(Chat)
            chat_container = chat_widget.query_one("#chat-container")
            
            system_chatbox = Chatbox(
                message=chat_message,
                model=self.chat_data.model,
                classes="system-message",
            )
            
            # Mount the chatbox to show it immediately
            await chat_container.mount(system_chatbox)
            
            # Scroll to the end
            chat_container.scroll_end(animate=False)
            
            # Add to database if chat has ID
            if self.chat_data.id:
                await self.chats_manager.add_message_to_chat(
                    chat_id=self.chat_data.id, 
                    message=chat_message
                )
            
        except Exception as e:
            # Fallback to notification if adding to chat fails
            self.app.notify(message, severity="information", timeout=8)
