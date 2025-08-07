"""
GraphRAG Manager for Elia Chat

This module handles integration with nano-graphrag for enhanced document
search and knowledge graph-based chat functionality.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

try:
    from nano_graphrag import GraphRAG, QueryParam
    from nano_graphrag.base import BaseKVStorage
    from nano_graphrag._utils import compute_args_hash
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

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

from elia_chat.config import GraphRAGConfig, EliaChatModel
from textual import log


class DocumentParser:
    """Handles parsing of various document formats."""
    
    @staticmethod
    def extract_text_from_txt(file_path: Path) -> str:
        """Extract text from .txt files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    @staticmethod
    def extract_text_from_md(file_path: Path) -> str:
        """Extract text from .md files."""
        return DocumentParser.extract_text_from_txt(file_path)
    
    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        """Extract text from .pdf files."""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
        text = ""
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            log.error(f"Error reading PDF {file_path}: {e}")
            return ""
        return text
    
    @staticmethod
    def extract_text_from_docx(file_path: Path) -> str:
        """Extract text from .docx files."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")
        
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            log.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path: Path) -> str:
        """Extract text from a file based on its extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return cls.extract_text_from_txt(file_path)
        elif suffix == '.md':
            return cls.extract_text_from_md(file_path)
        elif suffix == '.pdf':
            return cls.extract_text_from_pdf(file_path)
        elif suffix == '.docx':
            return cls.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


class GraphRAGManager:
    """Manages GraphRAG instances and document indexing."""
    
    def __init__(self, config: GraphRAGConfig):
        if not NANO_GRAPHRAG_AVAILABLE:
            raise ImportError("nano-graphrag is not available. Please install it first.")
        
        self.config = config
        self._graphrag_instance: Optional[GraphRAG] = None
        self._indexed_files: set[str] = set()
        
    @property
    def is_enabled(self) -> bool:
        """Check if GraphRAG is properly configured and enabled."""
        return (
            self.config.enabled and 
            self.config.storage_folder is not None and
            NANO_GRAPHRAG_AVAILABLE
        )
    
    def _get_graphrag_instance(self) -> GraphRAG:
        """Get or create a GraphRAG instance."""
        if self._graphrag_instance is None:
            if not self.config.storage_folder:
                raise ValueError("GraphRAG storage folder not configured")
            
            storage_path = Path(self.config.storage_folder)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Configure GraphRAG with custom models if specified
            kwargs = {
                "working_dir": str(storage_path),
                "enable_llm_cache": True,
            }
            
            # Add model configuration if specified
            if self.config.graphrag_model:
                # Here you would implement custom LLM function based on the model
                # For now, we'll use the default models
                pass
            
            self._graphrag_instance = GraphRAG(**kwargs)
            
        return self._graphrag_instance
    
    async def index_documents(self, force_reindex: bool = False) -> Dict[str, Any]:
        """Index documents from the configured documents folder."""
        if not self.is_enabled or not self.config.documents_folder:
            return {"error": "GraphRAG not properly configured"}
        
        documents_path = Path(self.config.documents_folder)
        if not documents_path.exists():
            return {"error": f"Documents folder does not exist: {documents_path}"}
        
        # Find supported document files
        supported_extensions = {'.txt', '.md', '.pdf', '.docx'}
        files_to_process = []
        
        for ext in supported_extensions:
            files_to_process.extend(documents_path.glob(f"**/*{ext}"))
        
        if not files_to_process:
            return {"message": "No supported documents found", "files_processed": 0}
        
        # Process files
        results = {"files_processed": 0, "files_skipped": 0, "errors": []}
        graphrag = self._get_graphrag_instance()
        
        for file_path in files_to_process:
            file_key = str(file_path.absolute())
            
            # Skip if already indexed (unless force_reindex)
            if not force_reindex and file_key in self._indexed_files:
                results["files_skipped"] += 1
                continue
            
            try:
                text_content = DocumentParser.extract_text(file_path)
                if text_content.strip():
                    # Use async insert if available
                    if hasattr(graphrag, 'ainsert'):
                        await graphrag.ainsert(text_content)
                    else:
                        # Run sync insert in executor
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, graphrag.insert, text_content)
                    
                    self._indexed_files.add(file_key)
                    results["files_processed"] += 1
                    log.info(f"Indexed document: {file_path.name}")
                else:
                    results["files_skipped"] += 1
                    log.warning(f"Document is empty: {file_path.name}")
                    
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {str(e)}"
                results["errors"].append(error_msg)
                log.error(error_msg)
        
        return results
    
    async def query_graphrag(self, query: str, mode: str = None) -> str:
        """Query the GraphRAG instance."""
        if not self.is_enabled:
            raise ValueError("GraphRAG is not enabled or configured")
        
        graphrag = self._get_graphrag_instance()
        query_mode = mode or self.config.query_mode
        
        try:
            param = QueryParam(mode=query_mode)
            
            # Use async query if available
            if hasattr(graphrag, 'aquery'):
                result = await graphrag.aquery(query, param=param)
            else:
                # Run sync query in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, graphrag.query, query, param)
            
            return result
            
        except Exception as e:
            log.error(f"Error querying GraphRAG: {e}")
            raise
    
    def get_indexed_files_count(self) -> int:
        """Get the number of indexed files."""
        return len(self._indexed_files)
    
    def clear_index(self) -> bool:
        """Clear the GraphRAG index."""
        if not self.config.storage_folder:
            return False
        
        try:
            storage_path = Path(self.config.storage_folder)
            if storage_path.exists():
                import shutil
                shutil.rmtree(storage_path)
                storage_path.mkdir(parents=True, exist_ok=True)
            
            self._graphrag_instance = None
            self._indexed_files.clear()
            return True
            
        except Exception as e:
            log.error(f"Error clearing GraphRAG index: {e}")
            return False


# Global instance
_graphrag_manager: Optional[GraphRAGManager] = None


def get_graphrag_manager(config: GraphRAGConfig) -> GraphRAGManager:
    """Get or create the global GraphRAG manager instance."""
    global _graphrag_manager
    if _graphrag_manager is None or _graphrag_manager.config != config:
        _graphrag_manager = GraphRAGManager(config)
    return _graphrag_manager


def is_graphrag_available() -> bool:
    """Check if nano-graphrag is available."""
    return NANO_GRAPHRAG_AVAILABLE
