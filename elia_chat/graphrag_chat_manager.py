"""
Enhanced Chat Manager with GraphRAG integration

This module extends the regular chat functionality to include
GraphRAG-based responses when enabled.
"""

import asyncio
from typing import AsyncGenerator, Optional
from textual import log

from elia_chat.models import ChatData, ChatMessage, EliaChatModel
from elia_chat.config import GraphRAGConfig
from elia_chat.graphrag_manager import get_graphrag_manager, is_graphrag_available

try:
    import litellm
    from litellm import acompletion
except ImportError:
    litellm = None
    acompletion = None


class GraphRAGChatManager:
    """Chat manager with GraphRAG integration."""
    
    def __init__(self, graphrag_config: GraphRAGConfig):
        self.graphrag_config = graphrag_config
        self.graphrag_manager = None
        
        if is_graphrag_available() and graphrag_config.enabled:
            self.graphrag_manager = get_graphrag_manager(graphrag_config)
    
    @property
    def is_graphrag_enabled(self) -> bool:
        """Check if GraphRAG is enabled and operational."""
        return (
            self.graphrag_manager is not None and 
            self.graphrag_manager.is_enabled
        )
    
    async def should_use_graphrag(self, user_message: str) -> bool:
        """Determine if a user message should use GraphRAG."""
        # For now, use GraphRAG for all queries when enabled
        # In the future, could add smarter detection
        return self.is_graphrag_enabled
    
    async def get_graphrag_context(self, user_message: str) -> Optional[str]:
        """Get relevant context from GraphRAG for a user message."""
        if not self.is_graphrag_enabled:
            return None
        
        try:
            context = await self.graphrag_manager.query_graphrag(
                user_message, 
                mode=self.graphrag_config.query_mode
            )
            return context
        except Exception as e:
            log.error(f"Error getting GraphRAG context: {e}")
            return None
    
    async def generate_enhanced_response(
        self,
        chat_data: ChatData,
        user_message: str,
        model: EliaChatModel,
    ) -> AsyncGenerator[str, None]:
        """Generate a response using GraphRAG context + LLM."""
        
        # Get GraphRAG context
        graphrag_context = await self.get_graphrag_context(user_message)
        
        if graphrag_context:
            # Enhance the system prompt with GraphRAG context
            enhanced_system_prompt = self._create_enhanced_system_prompt(
                chat_data.system_prompt.message["content"], 
                graphrag_context
            )
        else:
            # Fall back to regular chat
            enhanced_system_prompt = chat_data.system_prompt.message["content"]
        
        # Prepare messages for LLM
        messages = []
        
        # Add enhanced system prompt
        messages.append({
            "role": "system",
            "content": enhanced_system_prompt
        })
        
        # Add conversation history (excluding original system message)
        for chat_msg in chat_data.non_system_messages:
            messages.append(chat_msg.message)
        
        # Generate response using litellm
        if litellm is None:
            yield "Error: litellm not available for GraphRAG responses"
            return
        
        try:
            response = await acompletion(
                model=model.name,
                messages=messages,
                stream=True,
                temperature=model.temperature,
                api_key=model.api_key.get_secret_value() if model.api_key else None,
                api_base=str(model.api_base) if model.api_base else None,
                organization=model.organization,
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            log.error(f"Error generating enhanced response: {e}")
            yield f"Error generating response: {str(e)}"
    
    def _create_enhanced_system_prompt(self, original_prompt: str, graphrag_context: str) -> str:
        """Create an enhanced system prompt that includes GraphRAG context."""
        return f"""{original_prompt}

You have access to the following relevant information from the user's knowledge base:

---KNOWLEDGE BASE CONTEXT---
{graphrag_context}
---END CONTEXT---

When responding, use this context to provide more accurate and relevant answers. If the context is relevant to the user's question, incorporate it naturally into your response. If the context is not relevant, respond normally based on your training data."""
    
    async def generate_regular_response(
        self,
        chat_data: ChatData,
        model: EliaChatModel,
    ) -> AsyncGenerator[str, None]:
        """Generate a regular response without GraphRAG enhancement."""
        
        if litellm is None:
            yield "Error: litellm not available for responses"
            return
        
        # Prepare messages for LLM
        messages = []
        for chat_msg in chat_data.messages:
            messages.append(chat_msg.message)
        
        try:
            response = await acompletion(
                model=model.name,
                messages=messages,
                stream=True,
                temperature=model.temperature,
                api_key=model.api_key.get_secret_value() if model.api_key else None,
                api_base=str(model.api_base) if model.api_base else None,
                organization=model.organization,
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            log.error(f"Error generating regular response: {e}")
            yield f"Error generating response: {str(e)}"
