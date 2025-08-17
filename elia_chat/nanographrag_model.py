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
                        emit_line(f"[log] {msg}\n")
                bridge_handler = _BridgeHandler()
                bridge_handler.setLevel(_logging.INFO)
                bridge_handler.setFormatter(_logging.Formatter('%(levelname)s %(message)s'))
                bridge_logger.addHandler(bridge_handler)
                bridge_logger.propagate = True
            except Exception:
                bridge_handler = None

            try:
                if hasattr(session.graphrag_instance, 'ainsert'):
                    emit_line("Mode: async ainsert\n")
                    with capture():
                        await session.graphrag_instance.ainsert(text_content)
                        await flush()
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
            
            # Insert into GraphRAG
            if hasattr(session.graphrag_instance, 'ainsert'):
                await session.graphrag_instance.ainsert(text)
            else:
                # Run sync insert in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, session.graphrag_instance.insert, text)
            
            return {
                "message": "Successfully indexed text",
                "status": "success",
                "text_length": len(text)
            }
            
        except Exception as e:
            log.error(f"Error inserting text: {e}")
            return {"error": f"Failed to insert text: {str(e)}"}
    
    async def query_graphrag(self, chat_id: int, query: str, mode: Optional[str] = None) -> str:
        """Query the GraphRAG session."""
        session = self.get_or_create_session(chat_id)
        
        if not session.graphrag_instance:
            return "Error: GraphRAG not available for this session."
        
        try:
            query_mode = mode or self.default_query_mode
            param = QueryParam(mode=query_mode)
            
            # Query GraphRAG
            if hasattr(session.graphrag_instance, 'aquery'):
                result = await session.graphrag_instance.aquery(query, param=param)
            else:
                # Run sync query in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, session.graphrag_instance.query, query, param
                )
            
            return result
            
        except Exception as e:
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
                result = await self.insert_document(chat_id, doc_path)
                yield f"Document insertion result: {result.get('message', result.get('error', 'Unknown result'))}"
                return
            
            elif user_message.lower().startswith("/insert-text "):
                # Handle text insertion
                text_content = user_message[13:].strip()
                result = await self.insert_text(chat_id, text_content)
                yield f"Text insertion result: {result.get('message', result.get('error', 'Unknown result'))}"
                return
            
            elif user_message.lower().startswith("/query-mode "):
                # Handle query mode change
                new_mode = user_message[12:].strip()
                if new_mode in ["global", "local", "naive"]:
                    self.default_query_mode = new_mode
                    yield f"Query mode changed to: {new_mode}"
                else:
                    yield f"Invalid query mode. Available modes: global, local, naive"
                return
            
            elif user_message.lower() == "/help":
                # Show help
                help_text = """
**Nano-GraphRAG Commands:**

- `/insert <file_path>` - Insert a document (PDF, TXT, MD, DOCX) into the knowledge base
- `/insert-text <text>` - Insert raw text into the knowledge base  
- `/query-mode <mode>` - Change query mode (global, local, naive)
- `/status` - Show session status
- `/help` - Show this help message

**Query Modes:**
- `global` - Use global graph analysis (good for broad questions)
- `local` - Use local graph search (good for specific entities)
- `naive` - Use traditional RAG without graph structure

Regular messages will be answered using the knowledge base if available.
"""
                yield help_text
                return
            
            elif user_message.lower() == "/status":
                # Show session status
                session = self.get_or_create_session(chat_id)
                status_text = f"""
**Session Status:**
- Session ID: {session.session_id}
- Working Directory: {session.working_dir}
- GraphRAG Available: {session.graphrag_instance is not None}
- Indexed Documents: {len(session.indexed_documents)}
- Query Mode: {self.default_query_mode}
"""
                yield status_text
                return
            
            # Regular GraphRAG query - check if documents are available
            session = self.get_or_create_session(chat_id)
            
            if not session.indexed_documents:
                yield "❌ No documents have been inserted into this GraphRAG session yet.\n"
                yield "Please use `/insert <file_path>` to add documents before asking questions.\n"
                yield "Supported formats: .txt, .md, .pdf, .docx\n"
                yield "Example: `/insert C:\\path\\to\\document.txt`\n"
                return
            
            response = await self.query_graphrag(chat_id, user_message)
            
            # Stream the response in chunks
            chunk_size = 50  # Adjust as needed
            words = response.split()
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield chunk
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.05)
                
        except Exception as e:
            log.error(f"Error in stream_response: {e}")
            yield f"Error generating response: {str(e)}"
    
    async def generate_response(self, chat_data, user_message: str) -> AsyncGenerator[str, None]:
        """Generate a response using nano-graphrag - compatibility method for chat.py."""
        if not chat_data.id:
            yield "❌ Chat session not initialized properly.\n"
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
    
    # Create model configurations regardless of availability
    # The actual availability will be checked when models are used
    
    # Global GraphRAG model
    models.append(EliaChatModel(
        id="nano-graphrag-global",
        name="nano-graphrag-global", 
        display_name="Nano-GraphRAG (Global)",
        provider="Nano-GraphRAG",
        product="GraphRAG",
        description="Knowledge graph-powered chat with global analysis. Best for broad questions spanning multiple topics." + 
                   ("" if NANO_GRAPHRAG_AVAILABLE else " [Requires nano-graphrag installation]"),
        temperature=0.7,
    ))
    
    # Local GraphRAG model
    models.append(EliaChatModel(
        id="nano-graphrag-local",
        name="nano-graphrag-local",
        display_name="Nano-GraphRAG (Local)",
        provider="Nano-GraphRAG",
        product="GraphRAG",
        description="Knowledge graph-powered chat with local search. Best for specific entity-focused questions." +
                   ("" if NANO_GRAPHRAG_AVAILABLE else " [Requires nano-graphrag installation]"),
        temperature=0.7,
    ))
    
    # Naive RAG model
    models.append(EliaChatModel(
        id="nano-graphrag-naive",
        name="nano-graphrag-naive",
        display_name="Nano-GraphRAG (Naive)",
        provider="Nano-GraphRAG", 
        product="GraphRAG",
        description="Traditional RAG without graph structure. Good for simple document search." +
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
    return model_name.startswith("nano-graphrag-")
