import os
import tomllib
from pathlib import Path
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr
from typing import Dict, Any


class EliaChatModel(BaseModel):
    name: str
    """The name of the model e.g. `gpt-3.5-turbo`.
    This must match the name of the model specified by the provider.
    """
    id: str | None = None
    """If you have multiple versions of the same model (e.g. a personal
    OpenAI gpt-3.5 and a work OpenAI gpt-3.5 with different API keys/org keys),
    you need to be able to refer to them. For example, when setting the `default_model`
    key in the config, if you write `gpt-3.5`, there's no way to know whether you
    mean your work or your personal `gpt-3.5`. That's where `id` comes in."""
    display_name: str | None = None
    """The display name of the model in the UI."""
    provider: str | None = None
    """The provider of the model, e.g. openai, anthropic, etc"""
    api_key: SecretStr | None = None
    """If set, this will be used in place of the environment variable that
    would otherwise be used for this model (instead of OPENAI_API_KEY,
    ANTHROPIC_API_KEY, etc.)."""
    api_base: AnyHttpUrl | None = None
    """If set, this will be used as the base URL for making API calls.
    This can be useful if you're hosting models on a LocalAI server, for
    example."""
    organization: str | None = None
    """Some providers, such as OpenAI, allow you to specify an organization.
    In the case of """
    description: str | None = Field(default=None)
    """A description of the model which may appear inside the Elia UI."""
    product: str | None = Field(default=None)
    """For example `ChatGPT`, `Claude`, `Gemini`, etc."""
    temperature: float = Field(default=1.0)
    """The temperature to use. Low temperature means the same prompt is likely
    to produce similar results. High temperature means a flatter distribution
    when predicting the next token, and so the next token will be more random.
    Setting a very high temperature will likely produce junk output."""
    max_retries: int = Field(default=0)
    """The number of times to retry a request after it fails before giving up."""

    @property
    def lookup_key(self) -> str:
        return self.id or self.name


def get_builtin_openai_models() -> list[EliaChatModel]:
    return [
        EliaChatModel(
            id="elia-gpt-3.5-turbo",
            name="gpt-3.5-turbo",
            display_name="GPT-3.5 Turbo",
            provider="OpenAI",
            product="ChatGPT",
            description="Fast & inexpensive model for simple tasks.",
            temperature=0.7,
        ),
        EliaChatModel(
            id="elia-gpt-4o",
            name="gpt-4o",
            display_name="GPT-4o",
            provider="OpenAI",
            product="ChatGPT",
            description="Fastest and most affordable flagship model.",
            temperature=0.7,
        ),
        EliaChatModel(
            id="elia-gpt-4-turbo",
            name="gpt-4-turbo",
            display_name="GPT-4 Turbo",
            provider="OpenAI",
            product="ChatGPT",
            description="Previous high-intelligence model.",
            temperature=0.7,
        ),
    ]


def get_builtin_anthropic_models() -> list[EliaChatModel]:
    return [
        EliaChatModel(
            id="elia-claude-3-5-sonnet-20240620",
            name="claude-3-5-sonnet-20240620",
            display_name="Claude 3.5 Sonnet",
            provider="Anthropic",
            product="Claude 3.5",
            description=("Anthropic's most intelligent model"),
        ),
        EliaChatModel(
            id="elia-claude-3-haiku-20240307",
            name="claude-3-haiku-20240307",
            display_name="Claude 3 Haiku",
            provider="Anthropic",
            product="Claude 3",
            description=(
                "Fastest and most compact model for near-instant responsiveness"
            ),
        ),
        EliaChatModel(
            id="elia-claude-3-sonnet-20240229",
            name="claude-3-sonnet-20240229",
            display_name="Claude 3 Sonnet",
            provider="Anthropic",
            product="Claude 3",
            description=(
                "Ideal balance of intelligence and speed for enterprise workloads"
            ),
        ),
        EliaChatModel(
            id="elia-claude-3-opus-20240229",
            name="claude-3-opus-20240229",
            display_name="Claude 3 Opus",
            provider="Anthropic",
            product="Claude 3",
            description="Excels at writing and complex tasks",
        ),
    ]


def get_builtin_google_models() -> list[EliaChatModel]:
    return [
        EliaChatModel(
            id="elia-gemini/gemini-1.5-pro-latest",
            name="gemini/gemini-1.5-pro-latest",
            display_name="Gemini 1.5 Pro",
            provider="Google",
            product="Gemini",
            description="Excels at reasoning tasks including code and text generation, "
            "text editing, problem solving, data extraction and generation",
        ),
        EliaChatModel(
            id="elia-gemini/gemini-1.5-flash-latest",
            name="gemini/gemini-1.5-flash-latest",
            display_name="Gemini 1.5 Flash",
            provider="Google",
            product="Gemini",
            description="Fast and versatile performance across a variety of tasks",
        ),
    ]


def get_builtin_models() -> list[EliaChatModel]:
    models = (
        get_builtin_openai_models()
        + get_builtin_anthropic_models()
        + get_builtin_google_models()
        + get_builtin_nanographrag_models()
    )
    return models


class NanoGraphRAGConfig(BaseModel):
    """Configuration for Nano-GraphRAG integration."""
    
    enabled: bool = Field(default=False)
    """Whether Nano-GraphRAG models are enabled."""
    
    working_dir: str = Field(default="~/.elia/nanographrag")
    """Base directory for GraphRAG working directories."""
    
    openai_api_key: str | None = Field(default=None)
    """OpenAI API key for GraphRAG operations."""
    
    best_model: str = Field(default="gpt-4o")
    """Best model for GraphRAG operations (entity extraction, etc.)."""
    
    cheap_model: str = Field(default="gpt-4o-mini")
    """Cheaper model for GraphRAG operations (summarization, etc.)."""
    
    query_mode: str = Field(default="global")
    """Default query mode: global, local, or naive."""
    
    max_context_length: int = Field(default=4000)
    """Maximum context length for responses."""
    
    embedding_model: str | None = Field(default=None)
    """Custom embedding model to use."""


def get_builtin_nanographrag_models() -> list[EliaChatModel]:
    """Get built-in nano-graphrag models if available."""
    try:
        # Create a single unified nano-graphrag model
        # Query modes are handled within the chat session via commands
        models = []
        
        models.append(EliaChatModel(
            id="nano-graphrag",
            name="nano-graphrag", 
            display_name="Nano-GraphRAG",
            provider="Nano-GraphRAG",
            product="GraphRAG",
            description="Knowledge graph-powered chat with document indexing and multiple query modes. Use /help for commands.",
            temperature=0.7,
        ))
        
        return models
    except Exception as e:
        # Catch all exceptions to avoid breaking the model list
        import logging
        logging.debug(f"Failed to load nano-graphrag models: {e}")
    
    return []


def load_nanographrag_config() -> Dict[str, Any]:
    """Load nano-graphrag configuration from config.toml file."""
    from elia_chat.locations import config_directory
    
    config_file = config_directory() / "nanographrag.toml"
    
    if config_file.exists():
        try:
            with open(config_file, "rb") as f:
                config_data = tomllib.load(f)
            return config_data.get("nanographrag", {})
        except Exception:
            pass
    
    return {}


def save_nanographrag_config(config: Dict[str, Any]) -> bool:
    """Save nano-graphrag configuration to config.toml file."""
    from elia_chat.locations import config_directory
    
    config_directory().mkdir(parents=True, exist_ok=True)
    config_file = config_directory() / "nanographrag.toml"
    
    try:
        # Load existing config or create new
        existing_config = {}
        if config_file.exists():
            with open(config_file, "rb") as f:
                existing_config = tomllib.load(f)
        
        # Update nanographrag section
        existing_config["nanographrag"] = config
        
        # Write config file (we'll need tomli_w for writing)
        try:
            import tomli_w
            with open(config_file, "wb") as f:
                tomli_w.dump(existing_config, f)
        except ImportError:
            # Fallback to simple format
            lines = ["[nanographrag]"]
            for key, value in config.items():
                if isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                elif isinstance(value, bool):
                    lines.append(f'{key} = {str(value).lower()}')
                else:
                    lines.append(f'{key} = {value}')
            
            with open(config_file, "w") as f:
                f.write("\n".join(lines))
        
        return True
    except Exception:
        return False


class GraphRAGConfig(BaseModel):
    """Configuration for GraphRAG functionality."""
    
    enabled: bool = Field(default=False)
    """Whether GraphRAG is enabled for enhanced search and chat."""
    
    documents_folder: str | None = Field(default=None)
    """Folder path containing documents to be indexed (.txt, .md, .pdf files)."""
    
    storage_folder: str | None = Field(default=None)
    """Folder path for GraphRAG instance storage and persistence."""
    
    graphrag_model: str | None = Field(default=None)
    """Model to use for GraphRAG operations (entity extraction, etc.)."""
    
    embedding_model: str | None = Field(default=None)
    """Model to use for embeddings in GraphRAG."""
    
    query_mode: str = Field(default="global")
    """Default query mode for GraphRAG: 'global', 'local', or 'naive'."""


class LaunchConfig(BaseModel):
    """The config of the application at launch.

    Values may be sourced via command line options, env vars, config files.
    """

    model_config = ConfigDict(frozen=True)

    default_model: str = Field(default="elia-gpt-4o")
    """The ID or name of the default model."""
    system_prompt: str = Field(
        default=os.getenv(
            "ELIA_SYSTEM_PROMPT", "You are a helpful assistant named Elia."
        )
    )
    message_code_theme: str = Field(default="monokai")
    """The default Pygments syntax highlighting theme to be used in chatboxes."""
    models: list[EliaChatModel] = Field(default_factory=list)
    builtin_models: list[EliaChatModel] = Field(
        default_factory=get_builtin_models, init=False
    )
    theme: str = Field(default="nebula")
    graphrag: GraphRAGConfig = Field(default_factory=GraphRAGConfig)
    nanographrag: NanoGraphRAGConfig = Field(default_factory=NanoGraphRAGConfig)

    @property
    def all_models(self) -> list[EliaChatModel]:
        return self.models + self.builtin_models

    @property
    def default_model_object(self) -> EliaChatModel:
        from elia_chat.models import get_model

        return get_model(self.default_model, self)

    @classmethod
    def get_current(cls) -> "LaunchConfig":
        return cls()
