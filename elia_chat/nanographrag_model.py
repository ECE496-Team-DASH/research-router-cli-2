"""
Nano-GraphRAG Model Integration for Elia Chat

This module provides a specialized GraphRAG model that can be used as a chat model
in Elia, with per-session working directories and document insertion capabilities.
"""

import os
import uuid
import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any, Callable
import io
import contextlib
from dataclasses import dataclass

try:
    from nano_graphrag import GraphRAG, QueryParam
    NANO_GRAPHRAG_AVAILABLE = True
except ImportError:
    NANO_GRAPHRAG_AVAILABLE = False
    GraphRAG = None
    QueryParam = None

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from textual import log
from elia_chat.config import EliaChatModel


@dataclass
class GraphRAGSession:
    """Represents a GraphRAG session with its own working directory."""
    session_id: str
    working_dir: Path
    graphrag_instance: Optional[GraphRAG]
    indexed_documents: set[str]
    
    def __post_init__(self):
        """Initialize the session with proper directory structure."""
        # Create the working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a subdirectory for this specific session
        session_dir = self.working_dir / f"session_{self.session_id}"
        session_dir.mkdir(exist_ok=True, parents=True)
        
        # Update working_dir to the session-specific directory  
        self.working_dir = session_dir
        
        log.info(f"Created GraphRAG session directory: {self.working_dir}")


class NanoGraphRAGModel:
    """A GraphRAG-powered chat model for Elia."""
    
    def __init__(self, model_config: EliaChatModel, nanographrag_config: Dict[str, Any]):
        self.model_config = model_config
        self.config = nanographrag_config
        self.sessions: Dict[str, GraphRAGSession] = {}
        
        # Use current working directory for nanographrag storage instead of user data directory
        import os
        current_dir = Path(os.getcwd())
        nanographrag_dir = current_dir / "nanographrag_storage"
        self.base_working_dir = Path(nanographrag_config.get("working_dir", str(nanographrag_dir))).expanduser()
        self.base_working_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Using nano-graphrag storage directory: {self.base_working_dir}")
        
        # Query configuration
        self.default_query_mode = nanographrag_config.get("query_mode", "global")
        self.max_context_length = nanographrag_config.get("max_context_length", 4000)
        
        # Model configuration for GraphRAG
        self.best_model = nanographrag_config.get("best_model", "gpt-4o")
        self.cheap_model = nanographrag_config.get("cheap_model", "gpt-4o-mini")
        
    def get_or_create_session(self, chat_id: int) -> GraphRAGSession:
        """Get or create a GraphRAG session for a specific chat."""
        session_id = str(chat_id)
        
        if session_id not in self.sessions:
            session_working_dir = self.base_working_dir / f"chat_{session_id}"
            
            session = GraphRAGSession(
                session_id=session_id,
                working_dir=session_working_dir,
                graphrag_instance=None,
                indexed_documents=set()
            )
            
            # Initialize GraphRAG instance after session directory is created
            if NANO_GRAPHRAG_AVAILABLE:
                try:
                    log.info(f"Initializing GraphRAG for chat {chat_id} at {session.working_dir}")
                    
                    # Create a basic GraphRAG instance with minimal configuration
                    log.info("Creating GraphRAG instance with basic configuration...")
                    session.graphrag_instance = GraphRAG(
                        working_dir=str(session.working_dir),
                        enable_llm_cache=True,
                        enable_local=True,
                        enable_naive_rag=False,  # can toggle via config later
                        best_model_max_async=7,
                        cheap_model_max_async=7,
                    )
                    log.info(f"Successfully created GraphRAG session for chat {chat_id} at {session.working_dir}")
                except Exception as e:
                    log.error(f"Failed to create GraphRAG instance: {e}")
                    import traceback
                    log.error(f"Full traceback: {traceback.format_exc()}")
                    session.graphrag_instance = None
            else:
                log.error("nano-graphrag is not available - check installation")
                session.graphrag_instance = None
            
            self.sessions[session_id] = session
        
        return self.sessions[session_id]
    
    def _get_graphrag_kwargs(self) -> Dict[str, Any]:
        """Get configuration kwargs for GraphRAG initialization."""
        kwargs = {}
        
        # Add model configuration if available
        if self.config.get("openai_api_key"):
            os.environ["OPENAI_API_KEY"] = self.config["openai_api_key"]
        
        # Add other configuration options
        if "embedding_model" in self.config:
            kwargs["embedding_model"] = self.config["embedding_model"]
            
        return kwargs
    
    async def insert_document(self, chat_id: int, document_path: str, *, line_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Insert a document into the GraphRAG session.

        line_callback: optional function receiving each log/progress line (newline terminated).
        """
        import logging
        logger = logging.getLogger("nano-graphrag")

        def emit_line(txt: str):
            if line_callback:
                try:
                    line_callback(txt if txt.endswith("\n") else txt + "\n")
                except Exception:
                    pass

        emit_line("[insert] start")
        logger.info("=== INSERT DOCUMENT START ===")
        logger.info(f"Chat ID: {chat_id}")
        logger.info(f"Document path: {document_path}")

        session = self.get_or_create_session(chat_id)
        logger.info(f"Session obtained. GraphRAG instance available: {session.graphrag_instance is not None}")
        if not session.graphrag_instance:
            emit_line("GraphRAG instance not available\n")
            return {"error": "GraphRAG not available"}

        try:
            # Normalize path (strip quotes and whitespace)
            original_input = document_path
            document_path = document_path.strip().strip('"').strip("'")
            doc_path = Path(document_path).expanduser()
            if not doc_path.is_absolute():
                # Resolve relative to current working directory explicitly
                doc_path = Path(os.getcwd()) / doc_path
            try:
                doc_path = doc_path.resolve()
            except Exception:
                pass
            emit_line(f"[path] original input: {original_input}\n")
            emit_line(f"[path] normalized absolute: {doc_path}\n")
            emit_line(f"[session] working dir: {session.working_dir}\n")
            if doc_path.exists():
                try:
                    size = doc_path.stat().st_size
                    emit_line(f"[file] exists size={size} bytes\n")
                except Exception:
                    emit_line("[file] exists (size unknown)\n")
            else:
                emit_line("[file] NOT FOUND at normalized path\n")
            if not doc_path.exists():
                emit_line("Document not found\n")
                return {"error": f"Document not found: {document_path}"}

            doc_key = str(doc_path.absolute())
            if doc_key in session.indexed_documents:
                emit_line("Already indexed (skipping)\n")
                return {"message": "Document already indexed", "status": "skipped"}

            emit_line(f"Reading {doc_path.name}\n")
            from elia_chat.graphrag_manager import DocumentParser
            text_content = DocumentParser.extract_text(doc_path)
            emit_line(f"Content length: {len(text_content)} chars\n")
            if not text_content.strip():
                emit_line("Document empty after parsing\n")
                return {"error": "Document is empty or could not be parsed"}

            emit_line("Starting GraphRAG insertion...\n")

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            @contextlib.contextmanager
            def capture():
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    yield

            async def flush():
                for buf in (stdout_buf, stderr_buf):
                    data = buf.getvalue()
                    if data:
                        for line in data.splitlines():
                            if line.strip():
                                emit_line(line + "\n")
                        buf.truncate(0)
                        buf.seek(0)

            # Bridge nano-graphrag logger to callback for duration of insertion
            bridge_handler = None
            try:
                import logging as _logging
                bridge_logger = _logging.getLogger("nano-graphrag")
                
                class _BridgeHandler(_logging.Handler):
                    def emit(self, record):
                        msg = self.format(record)
                        # Filter out overly verbose debug messages and format appropriately
                        level = record.levelname
                        if level == "INFO":
                            emit_line(f"[log] {msg}\n")
                        elif level == "WARNING":
                            emit_line(f"[warning] {msg}\n")
                        elif level == "ERROR":
                            emit_line(f"[error] {msg}\n")
                        else:
                            emit_line(f"[{level.lower()}] {msg}\n")
                
                bridge_handler = _BridgeHandler()
                bridge_handler.setLevel(_logging.INFO)
                bridge_handler.setFormatter(_logging.Formatter('%(message)s'))
                
                # Remove any existing handlers to avoid duplicates
                for handler in bridge_logger.handlers[:]:
                    bridge_logger.removeHandler(handler)
                
                bridge_logger.addHandler(bridge_handler)
                bridge_logger.setLevel(_logging.INFO)
                bridge_logger.propagate = False  # Prevent propagation to avoid duplicate logs
            except Exception:
                bridge_handler = None

            try:
                if hasattr(session.graphrag_instance, 'ainsert'):
                    emit_line("Mode: async ainsert\n")
                    
                    # Create a progress monitoring task
                    progress_task = None
                    
                    async def progress_monitor():
                        """Monitor progress and emit periodic updates during long operations"""
                        steps = [
                            "Processing chunks...",
                            "Extracting entities...", 
                            "Building knowledge graph...",
                            "Generating community reports...",
                            "Finalizing storage..."
                        ]
                        step_idx = 0
                        while True:
                            await asyncio.sleep(3)  # Update every 3 seconds
                            if step_idx < len(steps):
                                emit_line(f"[progress] {steps[step_idx]}\n")
                                step_idx += 1
                            else:
                                emit_line("[progress] Still processing...\n")
                    
                    try:
                        # Start progress monitoring
                        progress_task = asyncio.create_task(progress_monitor())
                        
                        # Run the actual insertion with retry logic for rate limits
                        max_retries = 5
                        retry_delay = 10  # Start with 10 seconds
                        
                        for attempt in range(max_retries):
                            try:
                                with capture():
                                    await session.graphrag_instance.ainsert(text_content)
                                    await flush()
                                break  # Success - exit retry loop
                                
                            except Exception as e:
                                error_str = str(e)
                                if ("RateLimitError" in error_str or "rate limit" in error_str.lower() or 
                                    "quota" in error_str.lower() or "too many requests" in error_str.lower()):
                                    
                                    if attempt < max_retries - 1:
                                        emit_line(f"[RETRY] Rate limited, waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...\n")
                                        await asyncio.sleep(retry_delay)
                                        retry_delay = min(retry_delay * 2, 300)  # Exponential backoff, max 5 minutes
                                        continue
                                    else:
                                        emit_line(f"[ERROR] Rate limit exceeded after {max_retries} attempts\n")
                                        raise Exception(f"Rate limit exceeded after {max_retries} attempts. Please try again later.")
                                else:
                                    # Non-rate-limit error, don't retry
                                    raise
                        
                    finally:
                        # Stop progress monitoring
                        if progress_task:
                            progress_task.cancel()
                            try:
                                await progress_task
                            except asyncio.CancelledError:
                                pass
                else:
                    emit_line("Mode: sync insert (thread)\n")
                    loop = asyncio.get_event_loop()
                    def _run():
                        with capture():
                            session.graphrag_instance.insert(text_content)
                    await loop.run_in_executor(None, _run)
                    await flush()
                emit_line("Insertion finished\n")
            except Exception as insert_error:
                emit_line(f"Insertion error: {type(insert_error).__name__}: {insert_error}\n")
                # Attach brief traceback snippet for diagnostics
                import traceback
                tb_lines = traceback.format_exc().strip().splitlines()
                for tl in tb_lines[-6:]:  # last few lines
                    emit_line(f"[trace] {tl}\n")
                err_str = str(insert_error)
                if "EmptyNetworkError" in err_str:
                    emit_line("(Empty graph; marking as indexed)\n")
                else:
                    return {"error": f"Failed during insertion: {insert_error}"}
            finally:
                if bridge_handler:
                    try:
                        bridge_logger.removeHandler(bridge_handler)
                    except Exception:
                        pass

            session.indexed_documents.add(doc_key)
            emit_line(f"Indexed documents: {len(session.indexed_documents)}\n")
            return {"message": f"Successfully indexed {doc_path.name}", "status": "success", "document_count": len(session.indexed_documents)}
        except Exception as e:
            import traceback
            emit_line(f"Unexpected error: {type(e).__name__}: {e}\n")
            tb = traceback.format_exc().splitlines()
            for line in tb[-10:]:
                emit_line(f"[trace] {line}\n")
            return {"error": f"Failed to insert document: {type(e).__name__}: {str(e)}"}
    
    async def insert_text(self, chat_id: int, text: str) -> Dict[str, Any]:
        """Insert raw text into the GraphRAG session."""
        session = self.get_or_create_session(chat_id)
        
        if not session.graphrag_instance:
            return {"error": "GraphRAG not available"}
        
        try:
            if not text.strip():
                return {"error": "Text is empty"}
            
            # Insert into GraphRAG with retry logic
            max_retries = 5
            retry_delay = 10
            
            for attempt in range(max_retries):
                try:
                    if hasattr(session.graphrag_instance, 'ainsert'):
                        await session.graphrag_instance.ainsert(text)
                    else:
                        # Run sync insert in executor
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, session.graphrag_instance.insert, text)
                    break  # Success
                    
                except Exception as e:
                    error_str = str(e)
                    if ("RateLimitError" in error_str or "rate limit" in error_str.lower() or 
                        "quota" in error_str.lower() or "too many requests" in error_str.lower()):
                        
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay = min(retry_delay * 2, 300)
                            continue
                        else:
                            raise Exception(f"Rate limit exceeded after {max_retries} attempts. Please try again later.")
                    else:
                        raise
            
            return {
                "message": "Successfully indexed text",
                "status": "success",
                "text_length": len(text)
            }
            
        except Exception as e:
            log.error(f"Error inserting text: {e}")
            return {"error": f"Failed to insert text: {str(e)}"}
    
    async def query_graphrag(self, chat_id: int, query: str, mode: Optional[str] = None, *, line_callback: Optional[Callable[[str], None]] = None) -> str:
        """Query the GraphRAG session with optional log streaming."""
        session = self.get_or_create_session(chat_id)
        
        if not session.graphrag_instance:
            return "Error: GraphRAG not available for this session."
        
        def emit_line(txt: str):
            if line_callback:
                try:
                    line_callback(txt if txt.endswith("\n") else txt + "\n")
                except Exception:
                    pass
        
        try:
            query_mode = mode or self.default_query_mode
            param = QueryParam(mode=query_mode)
            
            emit_line(f"[QUERY] Starting {query_mode} query")
            emit_line(f"[QUERY] Question: {query}")
            
            # Set up logging capture for HTTP requests and other logs
            bridge_handler = None
            try:
                import logging as _logging
                
                # Capture logs from multiple sources
                loggers_to_capture = [
                    "nano-graphrag",
                    "httpx", 
                    "openai",
                    "requests",
                    "urllib3"
                ]
                
                class _QueryBridgeHandler(_logging.Handler):
                    def emit(self, record):
                        msg = self.format(record)
                        logger_name = record.name
                        level = record.levelname
                        
                        if "HTTP Request" in msg:
                            emit_line(f"[HTTP] {msg}")
                        elif level == "INFO":
                            emit_line(f"[{logger_name}] {msg}")
                        elif level == "WARNING":
                            emit_line(f"[WARNING] {msg}")
                        elif level == "ERROR":
                            emit_line(f"[ERROR] {msg}")
                        else:
                            emit_line(f"[{level}] {msg}")
                
                bridge_handler = _QueryBridgeHandler()
                bridge_handler.setLevel(_logging.INFO)
                bridge_handler.setFormatter(_logging.Formatter('%(message)s'))
                
                # Add handler to multiple loggers
                for logger_name in loggers_to_capture:
                    logger = _logging.getLogger(logger_name)
                    logger.addHandler(bridge_handler)
                    logger.setLevel(_logging.INFO)
                    
            except Exception:
                bridge_handler = None
            
            try:
                # Query GraphRAG
                if hasattr(session.graphrag_instance, 'aquery'):
                    emit_line("[QUERY] Using async query")
                    result = await session.graphrag_instance.aquery(query, param=param)
                else:
                    emit_line("[QUERY] Using sync query in executor")
                    # Run sync query in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, session.graphrag_instance.query, query, param
                    )
                
                emit_line("[QUERY] Query completed successfully")
                return result
                
            finally:
                # Clean up log handlers
                if bridge_handler:
                    try:
                        for logger_name in loggers_to_capture:
                            logger = _logging.getLogger(logger_name)
                            logger.removeHandler(bridge_handler)
                    except Exception:
                        pass
            
        except Exception as e:
            emit_line(f"[ERROR] Query failed: {str(e)}")
            log.error(f"Error querying GraphRAG: {e}")
            return f"Error querying knowledge base: {str(e)}"
    
    async def stream_response(self, chat_id: int, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using GraphRAG."""
        
        # Extract the last user message for GraphRAG query
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_message = content
                break
        
        if not user_message:
            yield "No user message found."
            return
        
        try:
            # Check if this is a special command
            if user_message.lower().startswith("/insert "):
                # Handle document insertion
                doc_path = user_message[8:].strip()
                
                # Start with initial message
                yield "# Document Insertion Starting\n\n[INIT] **Preparing to insert document...**\n\n"
                yield "**[PROGRESS] Insertion Progress:**\n\n```\n"
                
                # Use a queue to collect log lines for streaming
                log_queue = asyncio.Queue()
                
                def progress_callback(line: str):
                    if line.strip():
                        # Format log lines with simple text-based indicators
                        if line.startswith("[insert]"):
                            formatted_line = f"[INSERT] {line.strip()}"
                        elif line.startswith("[path]"):
                            formatted_line = f"[PATH] {line.strip()}"
                        elif line.startswith("[session]"):
                            formatted_line = f"[SESSION] {line.strip()}"
                        elif line.startswith("[file]"):
                            formatted_line = f"[FILE] {line.strip()}"
                        elif line.startswith("[log]"):
                            formatted_line = f"[LOG] {line.strip()}"
                        elif line.startswith("[warning]"):
                            formatted_line = f"[WARNING] {line.strip()}"
                        elif line.startswith("[error]"):
                            formatted_line = f"[ERROR] {line.strip()}"
                        elif line.startswith("[progress]"):
                            formatted_line = f"[STEP] {line.strip()}"
                        elif line.startswith("[trace]"):
                            formatted_line = f"[TRACE] {line.strip()}"
                        elif "Insertion finished" in line:
                            formatted_line = f"[SUCCESS] {line.strip()}"
                        elif "Starting GraphRAG insertion" in line:
                            formatted_line = f"[START] {line.strip()}"
                        elif "Reading" in line:
                            formatted_line = f"[READ] {line.strip()}"
                        elif "Content length:" in line:
                            formatted_line = f"[DATA] {line.strip()}"
                        elif "Entity Extraction" in line:
                            formatted_line = f"[EXTRACT] {line.strip()}"
                        elif "Community Report" in line:
                            formatted_line = f"[REPORT] {line.strip()}"
                        elif "inserting" in line.lower():
                            formatted_line = f"[STORE] {line.strip()}"
                        elif "Mode:" in line:
                            formatted_line = f"[MODE] {line.strip()}"
                        else:
                            formatted_line = f"[INFO] {line.strip()}"
                        
                        # Put the formatted line in the queue (non-blocking)
                        try:
                            log_queue.put_nowait(formatted_line)
                        except asyncio.QueueFull:
                            pass  # Skip if queue is full
                
                # Create an async task for the document insertion
                async def insertion_task():
                    return await self.insert_document(chat_id, doc_path, line_callback=progress_callback)
                
                # Start the insertion task
                task = asyncio.create_task(insertion_task())
                
                # Stream log lines as they come in
                while not task.done():
                    try:
                        # Wait for log lines with a short timeout
                        log_line = await asyncio.wait_for(log_queue.get(), timeout=0.1)
                        yield f"{log_line}\n"
                    except asyncio.TimeoutError:
                        # No log line available, continue checking if task is done
                        continue
                    except Exception:
                        # Any other error, break the loop
                        break
                
                # Drain any remaining log lines
                while not log_queue.empty():
                    try:
                        log_line = log_queue.get_nowait()
                        yield f"{log_line}\n"
                    except asyncio.QueueEmpty:
                        break
                
                # Close the code block
                yield "```\n"
                
                # Get the result
                try:
                    result = await task
                except Exception as e:
                    result = {"error": f"Failed to insert document: {str(e)}"}
                
                yield "\n---\n\n"
                
                # Format the result with progress
                if result.get('status') == 'success':
                    response = f"""
# Document Insertion Successful

[SUCCESS] **{result.get('message', 'Document indexed successfully')}**

- **Total indexed documents:** {result.get('document_count', 'Unknown')}
- **Session:** {chat_id}

You can now ask questions using:
- Regular messages (uses current mode: **{self.default_query_mode}**)
- `/query <mode> <question>` for specific modes
- `/help` for more commands
"""
                elif result.get('status') == 'skipped':
                    response = f"""
# Document Already Indexed

[SKIPPED] **{result.get('message', 'Document was already in the knowledge base')}**

- **Total indexed documents:** {result.get('document_count', 'Unknown')}
- **Session:** {chat_id}
"""
                else:
                    response = f"""
# Document Insertion Failed

[ERROR] **Error:** {result.get('error', 'Unknown error occurred')}

Please check:
- File path is correct and accessible
- File format is supported (.txt, .md, .pdf, .docx)
- You have read permissions for the file

Example: `/insert C:\\path\\to\\document.txt`
"""
                
                yield response
                return
            
            elif user_message.lower().startswith("/insert-text "):
                # Handle text insertion
                text_content = user_message[13:].strip()
                result = await self.insert_text(chat_id, text_content)
                
                if result.get('status') == 'success':
                    response = f"""
# Text Insertion Successful

[SUCCESS] **Text content indexed successfully**

- **Text length:** {result.get('text_length', 'Unknown')} characters
- **Session:** {chat_id}

You can now ask questions about the inserted text using:
- Regular messages (uses current mode: **{self.default_query_mode}**)
- `/query <mode> <question>` for specific modes
"""
                else:
                    response = f"""
# Text Insertion Failed

[ERROR] **Error:** {result.get('error', 'Unknown error occurred')}

Please check:
- Text content is not empty
- Text is properly formatted

Example: `/insert-text This is important information about the project.`
"""
                
                yield response
                return
            
            elif user_message.lower().startswith("/insert-folder "):
                # Handle folder insertion
                folder_path = user_message[15:].strip()
                
                yield "# Folder Insertion Starting\n\n[INIT] **Preparing to insert documents from folder...**\n\n"
                yield f"**Folder:** `{folder_path}`\n\n"
                yield "**[PROGRESS] Scanning for documents:**\n\n```\n"
                
                try:
                    from pathlib import Path
                    import os
                    folder = Path(folder_path.strip().strip('"').strip("'")).expanduser()
                    if not folder.is_absolute():
                        folder = Path(os.getcwd()) / folder
                    
                    if not folder.exists() or not folder.is_dir():
                        yield "```\n\n❌ **Error:** Folder not found or invalid path\n"
                        return
                    
                    # Find supported files
                    supported_extensions = {'.txt', '.md', '.pdf', '.docx'}
                    files_found = []
                    for ext in supported_extensions:
                        files_found.extend(folder.rglob(f"*{ext}"))
                    
                    yield f"Found {len(files_found)} supported documents\n"
                    
                    if not files_found:
                        yield "```\n\n⚠️ **No supported documents found in folder**\n"
                        return
                    
                    yield "```\n\n**[PROGRESS] Processing documents:**\n\n"
                    
                    successful = 0
                    skipped = 0
                    failed = 0
                    
                    for i, file_path in enumerate(files_found, 1):
                        yield f"📄 [{i}/{len(files_found)}] {file_path.name}\n"
                        
                        try:
                            result = await self.insert_document(chat_id, str(file_path))
                            
                            if "error" in result:
                                yield f"   ❌ Error: {result['error']}\n"
                                failed += 1
                            elif result.get("status") == "skipped":
                                yield f"   ⏭️ Skipped (already indexed)\n"
                                skipped += 1
                            else:
                                yield f"   ✅ Success\n"
                                successful += 1
                        except Exception as e:
                            yield f"   ❌ Exception: {str(e)}\n"
                            failed += 1
                        
                        await asyncio.sleep(0.05)  # Brief pause for responsiveness
                    
                    yield f"\n---\n\n"
                    yield f"# Folder Insertion Complete\n\n"
                    yield f"**Summary:**\n"
                    yield f"- ✅ Successfully indexed: {successful}\n"
                    yield f"- ⏭️ Skipped (duplicates): {skipped}\n"
                    yield f"- ❌ Failed: {failed}\n"
                    yield f"- 📁 Total processed: {len(files_found)}\n\n"
                    
                    if successful > 0:
                        session_info = self.get_session_info(chat_id)
                        yield f"📚 Total documents in session: {session_info.get('indexed_documents', 0)}\n\n"
                        yield "🎉 Folder insertion completed! You can now ask questions about the content.\n"
                    
                except Exception as e:
                    yield f"```\n\n❌ **Error during folder insertion:** {str(e)}\n"
                
                return
            
            elif user_message.lower().startswith("/insert-arxiv "):
                # Handle arXiv search and insertion with paper selection
                search_query = user_message[14:].strip()
                
                yield "# arXiv Paper Search & Selection\n\n[INIT] **Searching arXiv database...**\n\n"
                yield f"**Query:** `{search_query}`\n\n"
                
                try:
                    # Check if feedparser is available
                    try:
                        import feedparser
                        import aiohttp
                        import tempfile
                        from pathlib import Path
                    except ImportError as e:
                        yield f"[ERROR] **Error:** Missing required package: {str(e)}\n"
                        yield "Install with: `pip install feedparser aiohttp`\n"
                        return
                    
                    yield "**[PROGRESS] Searching papers:**\n\n```\n"
                    
                    # Search arXiv
                    base_url = "http://export.arxiv.org/api/query"
                    params = {
                        "search_query": search_query,
                        "start": 0,
                        "max_results": 10,  # More papers to choose from
                        "sortBy": "relevance",
                        "sortOrder": "descending"
                    }
                    
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
                                
                                yield f"Found {len(papers)} papers with PDFs\n"
                                yield "```\n\n"
                                
                                if not papers:
                                    yield "[WARNING] **No papers with PDFs found for this query**\n"
                                    return
                                
                                # Display papers for selection
                                yield "## [PAPERS] Found Papers - Choose which to insert:\n\n"
                                yield "*Reply with the paper numbers you want to insert (e.g., '1,3,5' or 'all')*\n\n"
                                
                                for i, paper in enumerate(papers, 1):
                                    yield f"**{i}.** {paper['title']}\n"
                                    yield f"   [AUTHORS] **Authors:** {paper['authors_str']}\n"
                                    yield f"   [ID] **arXiv ID:** {paper['arxiv_id']}\n"
                                    yield f"   [DATE] **Published:** {paper['published'][:10]}\n"
                                    yield f"   [SUMMARY] **Summary:** {paper['summary']}\n\n"
                                
                                yield "---\n\n"
                                yield "[INFO] **Instructions:**\n"
                                yield "- Type paper numbers separated by commas: `/insert-selected 1,3,5`\n"
                                yield "- Or insert all papers: `/insert-selected all`\n"
                                yield "- Or search again with a different query\n\n"
                                
                                # Store papers in session for later selection
                                session = self.get_or_create_session(chat_id)
                                session.pending_arxiv_papers = papers
                                
                            else:
                                yield f"```\n\n[ERROR] **Error:** Failed to search arXiv (HTTP {response.status})\n"
                
                except Exception as e:
                    yield f"[ERROR] **Error during arXiv search:** {str(e)}\n"
                
                return
            
            elif user_message.lower().startswith("/insert-selected "):
                # Handle selected paper insertion
                selection = user_message[16:].strip().lower()
                
                session = self.get_or_create_session(chat_id)
                if not hasattr(session, 'pending_arxiv_papers') or not session.pending_arxiv_papers:
                    yield "[ERROR] **No papers available for selection.** Use `/insert-arxiv <query>` first.\n"
                    return
                
                papers = session.pending_arxiv_papers
                selected_papers = []
                
                try:
                    if selection == "all":
                        selected_papers = papers
                        yield f"# Inserting All {len(papers)} Papers\n\n"
                    else:
                        # Parse selected numbers
                        indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip().isdigit()]
                        selected_papers = [papers[i] for i in indices if 0 <= i < len(papers)]
                        
                        if not selected_papers:
                            yield "[ERROR] **Invalid selection.** Please use numbers like '1,3,5' or 'all'.\n"
                            return
                        
                        yield f"# Inserting {len(selected_papers)} Selected Papers\n\n"
                    
                    # Import requirements for download and insertion
                    try:
                        import aiohttp
                        import tempfile
                        from pathlib import Path
                    except ImportError as e:
                        yield f"[ERROR] **Error:** Missing required package: {str(e)}\n"
                        return
                    
                    yield "**[PROGRESS] Downloading papers to current directory:**\n\n"
                    
                    # Download to current working directory
                    import os
                    current_dir = Path(os.getcwd())
                    downloads_dir = current_dir / "arxiv_papers"
                    downloads_dir.mkdir(exist_ok=True)
                    
                    yield f"[INFO] **Download folder:** `{downloads_dir}`\n\n"
                    
                    successful = 0
                    failed = 0
                    
                    async with aiohttp.ClientSession() as session_http:
                        for i, paper in enumerate(selected_papers, 1):
                            # Progress indicator
                            progress_percent = int((i-1) / len(selected_papers) * 100)
                            progress_bar = "=" * (progress_percent // 10) + "-" * (10 - progress_percent // 10)
                            yield f"[PROGRESS] **Progress: [{progress_bar}] {progress_percent}%**\n\n"
                            yield f"[PAPER] **[{i}/{len(selected_papers)}] {paper['title'][:50]}...**\n"
                            
                            try:
                                # Download PDF to session folder
                                filename = f"{paper['arxiv_id'].replace('/', '_')}.pdf"
                                download_path = downloads_dir / filename
                                
                                # Skip if already downloaded
                                if download_path.exists():
                                    yield f"   [SKIP] Already downloaded: {filename}\n"
                                    successful += 1
                                    continue
                                
                                yield f"   [DOWNLOAD] Downloading PDF...\n"
                                async with session_http.get(paper['pdf_url']) as pdf_response:
                                    if pdf_response.status == 200:
                                        content = await pdf_response.read()
                                        with open(download_path, 'wb') as f:
                                            f.write(content)
                                        
                                        yield f"   [OK] Downloaded ({len(content)//1024} KB)\n"
                                        
                                        successful += 1
                                    else:
                                        yield f"   [ERROR] Failed to download PDF (HTTP {pdf_response.status})\n"
                                        failed += 1
                            
                            except Exception as e:
                                yield f"   [ERROR] Exception: {str(e)}\n"
                                failed += 1
                            
                            yield f"\n"
                            # Small delay between downloads
                            if i < len(selected_papers):
                                await asyncio.sleep(1)
                    
                    
                    # Clear pending papers
                    session.pending_arxiv_papers = None
                    
                    yield f"---\n\n"
                    yield f"# [COMPLETE] Download Complete\n\n"
                    yield f"**Summary:**\n"
                    yield f"- [OK] Successfully downloaded: **{successful}** papers\n"
                    yield f"- [ERROR] Failed: **{failed}** papers\n"
                    yield f"- [TOTAL] Total selected: **{len(selected_papers)}** papers\n\n"
                    yield f"[FOLDER] **Papers saved to:** `{downloads_dir}`\n\n"
                    
                    if successful > 0:
                        yield f"**[NEXT] Use F9 to insert papers into knowledge base:**\n"
                        yield f"- Insert folder: `{downloads_dir}`\n"
                        yield f"- Or insert individual files from the folder\n\n"
                    
                    yield f"**Download complete!** Papers ready for manual insertion.\n"
                
                except Exception as e:
                    yield f"[ERROR] **Error during paper insertion:** {str(e)}\n"
                
                return
            
            elif user_message.lower() == "/list-docs":
                # List all indexed documents
                session = self.get_or_create_session(chat_id)
                
                yield "# Indexed Documents\n\n"
                
                if not session.indexed_documents:
                    yield "📂 **No documents indexed in this session**\n\n"
                    yield "Use `/insert <file>`, `/insert-folder <folder>`, or **F9** to add documents.\n"
                else:
                    yield f"📚 **Total indexed documents:** {len(session.indexed_documents)}\n\n"
                    yield "**Document List:**\n\n"
                    
                    for i, doc_path in enumerate(sorted(session.indexed_documents), 1):
                        doc_name = Path(doc_path).name
                        yield f"{i}. `{doc_name}`\n"
                        yield f"   Path: `{doc_path}`\n\n"
                    
                    session_info = self.get_session_info(chat_id)
                    yield f"📂 **Session storage:** `{session_info.get('working_dir', 'Unknown')}`\n"
                    yield f"🔧 **Current query mode:** `{self.default_query_mode}`\n"
                
                return
            
            elif user_message.lower().startswith("/query "):
                # Handle query with specific mode
                query_parts = user_message[7:].strip().split(" ", 1)
                if len(query_parts) >= 2:
                    mode = query_parts[0].lower()
                    query_text = query_parts[1]
                    if mode in ["global", "local", "naive"]:
                        # Set up streaming logs for the query
                        yield f"# Query Response ({mode.title()} Mode)\n\n"
                        yield "**[QUERY] Processing:**\n\n```\n"
                        
                        # Use a queue to collect log lines for streaming
                        query_log_queue = asyncio.Queue()
                        
                        def query_progress_callback(line: str):
                            if line.strip():
                                # Format query log lines
                                formatted_line = line.strip()
                                try:
                                    query_log_queue.put_nowait(formatted_line)
                                except asyncio.QueueFull:
                                    pass
                        
                        # Create an async task for the query
                        async def query_task():
                            return await self.query_graphrag(chat_id, query_text, mode, line_callback=query_progress_callback)
                        
                        # Start the query task
                        task = asyncio.create_task(query_task())
                        
                        # Stream log lines as they come in
                        while not task.done():
                            try:
                                # Wait for log lines with a short timeout
                                log_line = await asyncio.wait_for(query_log_queue.get(), timeout=0.1)
                                yield f"{log_line}\n"
                            except asyncio.TimeoutError:
                                # No log line available, continue checking if task is done
                                continue
                            except Exception:
                                # Any other error, break the loop
                                break
                        
                        # Drain any remaining log lines
                        while not query_log_queue.empty():
                            try:
                                log_line = query_log_queue.get_nowait()
                                yield f"{log_line}\n"
                            except asyncio.QueueEmpty:
                                break
                        
                        # Close the log block and get the result
                        yield "```\n\n"
                        
                        try:
                            response = await task
                        except Exception as e:
                            yield f"[ERROR] Query failed: {str(e)}\n"
                            return
                        
                        # Stream the response in chunks with proper formatting
                        yield "**[RESPONSE] Answer:**\n\n"
                        lines = response.split('\n')
                        current_line = ""
                        
                        for line in lines:
                            current_line += line + "\n"
                            # Check if we have enough content to yield
                            if len(current_line) >= 100 or line.strip() == "" or line.startswith('#') or line.startswith('-') or line.startswith('*'):
                                if current_line.strip():
                                    yield current_line
                                    current_line = ""
                                    await asyncio.sleep(0.1)
                        
                        # Yield any remaining content
                        if current_line.strip():
                            yield current_line
                    else:
                        yield f"Invalid query mode. Available modes: global, local, naive"
                else:
                    yield "Usage: /query <mode> <question>\nExample: /query global What are the main topics?"
                return
            
            elif user_message.lower().startswith("/mode "):
                # Handle query mode change
                new_mode = user_message[6:].strip()
                if new_mode in ["global", "local", "naive"]:
                    old_mode = self.default_query_mode
                    self.default_query_mode = new_mode
                    response = f"""
# Query Mode Changed

[MODE] **Default query mode updated**

- **Previous mode:** {old_mode}
- **New mode:** {new_mode}

## Mode Descriptions
- **global** - Broad analysis across all documents
- **local** - Entity-focused search  
- **naive** - Traditional document search

Regular messages will now use the **{new_mode}** mode by default.
Use `/query <mode> <question>` to override for specific queries.
"""
                else:
                    response = f"""
# Invalid Query Mode

[ERROR] **Error:** `{new_mode}` is not a valid query mode.

## Available Modes
- **global** - Broad analysis across all documents
- **local** - Entity-focused search
- **naive** - Traditional document search

**Usage:** `/mode <global|local|naive>`
**Example:** `/mode global`
"""
                
                yield response
                return
            
            elif user_message.lower() == "/help":
                # Show help
                help_text = """
# Nano-GraphRAG Commands

## Document Management
- `/insert <file_path>` - Insert a single document (PDF, TXT, MD, DOCX) into the knowledge base
- `/insert-folder <folder_path>` - Insert all supported documents from a folder
- `/insert-arxiv <search_query>` - Search arXiv papers and select which to download
- `/insert-selected <numbers>` - Download papers to current directory (e.g., '1,3,5' or 'all')
- `/insert-text <text>` - Insert raw text into the knowledge base

## Enhanced Insertion (via F9 key)
- Press **F9** to open the enhanced document insertion interface with:
  - File browser with directory tree
  - Bulk folder insertion with filters
  - arXiv paper search and download
  - Duplicate detection options
  - File type filtering

## Query Commands
- `/query <mode> <question>` - Query with specific mode (global/local/naive)
- `/mode <mode>` - Change default query mode for regular messages
- `/status` - Show session status and indexed documents
- `/list-docs` - List all indexed documents in current session
- `/help` - Show this help message

## Query Modes
- **global** - Use global graph analysis (good for broad questions spanning topics)
- **local** - Use local graph search (good for specific entity-focused questions)
- **naive** - Use traditional RAG without graph structure (simple document search)

## Examples
- `/insert C:\\Documents\\research.pdf`
- `/insert-folder C:\\Documents\\Papers\\`
- `/insert-arxiv machine learning transformers` (search papers)
- `/insert-selected 1,3,5` (download to current directory)
- `/query global What are the main themes in the documents?`
- `/query local Tell me about John Smith`
- `/mode global`

Regular messages will be answered using the current default mode if documents are available.
**Press F9 for the enhanced insertion interface with all advanced features!**
"""
                yield help_text
                return
            
            elif user_message.lower() == "/status":
                # Show session status
                session = self.get_or_create_session(chat_id)
                status_text = f"""
# Session Status

- **Session ID:** {session.session_id}
- **Working Directory:** {session.working_dir}
- **GraphRAG Available:** {session.graphrag_instance is not None}
- **Indexed Documents:** {len(session.indexed_documents)}
- **Current Query Mode:** {self.default_query_mode}

## Available Query Modes
- **global** - Broad analysis across all documents
- **local** - Entity-focused search
- **naive** - Traditional document search
"""
                yield status_text
                return
            
            # Regular GraphRAG query - check if documents are available
            session = self.get_or_create_session(chat_id)
            
            if not session.indexed_documents:
                no_docs_message = """
# No Documents Available

[EMPTY] **No documents have been inserted into this GraphRAG session yet.**

To get started:
1. Use `/insert <file_path>` to add documents to the knowledge base
2. Supported formats: `.txt`, `.md`, `.pdf`, `.docx`
3. Example: `/insert C:\\path\\to\\document.txt`

Once documents are indexed, you can:
- Ask questions using the current query mode (**{mode}**)
- Use `/query <mode> <question>` for specific query modes
- Use `/help` for more commands
""".format(mode=self.default_query_mode)
                yield no_docs_message
                return
            
            # Set up streaming logs for the regular query
            yield f"# Query Response ({self.default_query_mode.title()} Mode)\n\n"
            yield "**[QUERY] Processing:**\n\n```\n"
            
            # Use a queue to collect log lines for streaming
            query_log_queue = asyncio.Queue()
            
            def query_progress_callback(line: str):
                if line.strip():
                    # Format query log lines
                    formatted_line = line.strip()
                    try:
                        query_log_queue.put_nowait(formatted_line)
                    except asyncio.QueueFull:
                        pass
            
            # Create an async task for the query
            async def query_task():
                return await self.query_graphrag(chat_id, user_message, line_callback=query_progress_callback)
            
            # Start the query task
            task = asyncio.create_task(query_task())
            
            # Stream log lines as they come in
            while not task.done():
                try:
                    # Wait for log lines with a short timeout
                    log_line = await asyncio.wait_for(query_log_queue.get(), timeout=0.1)
                    yield f"{log_line}\n"
                except asyncio.TimeoutError:
                    # No log line available, continue checking if task is done
                    continue
                except Exception:
                    # Any other error, break the loop
                    break
            
            # Drain any remaining log lines
            while not query_log_queue.empty():
                try:
                    log_line = query_log_queue.get_nowait()
                    yield f"{log_line}\n"
                except asyncio.QueueEmpty:
                    break
            
            # Close the log block and get the result
            yield "```\n\n"
            
            try:
                response = await task
            except Exception as e:
                yield f"[ERROR] Query failed: {str(e)}\n"
                return
            
            # Stream the response in chunks with proper formatting
            yield "**[RESPONSE] Answer:**\n\n"
            lines = response.split('\n')
            current_line = ""
            
            for line in lines:
                current_line += line + "\n"
                # Check if we have enough content to yield (about 100-200 chars)
                if len(current_line) >= 100 or line.strip() == "" or line.startswith('#') or line.startswith('-') or line.startswith('*'):
                    if current_line.strip():
                        yield current_line
                        current_line = ""
                        await asyncio.sleep(0.1)  # Slightly longer delay for better readability
            
            # Yield any remaining content
            if current_line.strip():
                yield current_line
                
        except Exception as e:
            log.error(f"Error in stream_response: {e}")
            yield f"Error generating response: {str(e)}"
    
    async def generate_response(self, chat_data, user_message: str) -> AsyncGenerator[str, None]:
        """Generate a response using nano-graphrag - compatibility method for chat.py."""
        if not chat_data.id:
            yield "[ERROR] Chat session not initialized properly.\n"
            return
            
        # Convert to the expected message format and delegate to stream_response
        messages = [{"role": "user", "content": user_message}]
        async for chunk in self.stream_response(chat_data.id, messages):
            yield chunk
    
    def get_session_info(self, chat_id: int) -> Dict[str, Any]:
        """Get information about a GraphRAG session."""
        session = self.get_or_create_session(chat_id)
        
        return {
            "session_id": session.session_id,
            "working_dir": str(session.working_dir),
            "graphrag_available": session.graphrag_instance is not None,
            "indexed_documents": len(session.indexed_documents),
            "query_mode": self.default_query_mode
        }
    
    def cleanup_session(self, chat_id: int) -> bool:
        """Clean up a GraphRAG session."""
        session_id = str(chat_id)
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


def create_nanographrag_models(nanographrag_config: Dict[str, Any]) -> list[EliaChatModel]:
    """Create nano-graphrag model configurations."""
    models = []
    
    # Create single unified model configuration
    # Query modes are handled within the chat session via commands
    
    models.append(EliaChatModel(
        id="nano-graphrag",
        name="nano-graphrag", 
        display_name="Nano-GraphRAG",
        provider="Nano-GraphRAG",
        product="GraphRAG",
        description="Knowledge graph-powered chat with document indexing and multiple query modes. Use /help for commands." + 
                   ("" if NANO_GRAPHRAG_AVAILABLE else " [Requires nano-graphrag installation]"),
        temperature=0.7,
    ))
    
    return models


# Global registry for GraphRAG models
_graphrag_models: Dict[str, NanoGraphRAGModel] = {}


def get_nanographrag_model(model_name: str, nanographrag_config: Dict[str, Any]) -> Optional[NanoGraphRAGModel]:
    """Get or create a nano-graphrag model instance."""
    if model_name not in _graphrag_models:
        # Create model config
        model_config = EliaChatModel(
            name=model_name,
            display_name=f"Nano-GraphRAG ({model_name.split('-')[-1].title()})",
            provider="Nano-GraphRAG"
        )
        
        _graphrag_models[model_name] = NanoGraphRAGModel(model_config, nanographrag_config)
    
    return _graphrag_models[model_name]


def is_nanographrag_model(model_name: str) -> bool:
    """Check if a model name is a nano-graphrag model."""
    return model_name == "nano-graphrag"
