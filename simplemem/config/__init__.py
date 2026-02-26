"""SimpleMem config compatibility layer for legacy module layout."""

from dataclasses import dataclass
import config as _legacy_config


@dataclass(slots=True)
class SimpleMemConfig:
    openai_api_key: str = getattr(_legacy_config, "OPENAI_API_KEY", "")
    openai_base_url: str | None = getattr(_legacy_config, "OPENAI_BASE_URL", None)
    llm_model: str = getattr(_legacy_config, "LLM_MODEL", "")
    llm_temperature: float = float(getattr(_legacy_config, "LLM_TEMPERATURE", 0.1))
    llm_top_p: float = float(getattr(_legacy_config, "LLM_TOP_P", 1.0))
    top_p: float = float(getattr(_legacy_config, "LLM_TOP_P", 1.0))
    embedding_base_url: str | None = getattr(_legacy_config, "EMBEDDING_BASE_URL", None)
    embedding_api_key: str | None = getattr(_legacy_config, "EMBEDDING_API_KEY", None)
    embedding_model: str = getattr(_legacy_config, "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    embedding_dimension: int = int(getattr(_legacy_config, "EMBEDDING_DIMENSION", 1024))
    embedding_context_length: int = int(getattr(_legacy_config, "EMBEDDING_CONTEXT_LENGTH", 32768))
    window_size: int = int(getattr(_legacy_config, "WINDOW_SIZE", 40))
    overlap_size: int = int(getattr(_legacy_config, "OVERLAP_SIZE", 2))
    semantic_top_k: int = int(getattr(_legacy_config, "SEMANTIC_TOP_K", 25))
    keyword_top_k: int = int(getattr(_legacy_config, "KEYWORD_TOP_K", 5))
    structured_top_k: int = int(getattr(_legacy_config, "STRUCTURED_TOP_K", 5))
    lancedb_path: str = getattr(_legacy_config, "LANCEDB_PATH", "./lancedb_data")
    memory_table_name: str = getattr(_legacy_config, "MEMORY_TABLE_NAME", "memory_entries")
    enable_thinking: bool = bool(getattr(_legacy_config, "ENABLE_THINKING", False))
    use_streaming: bool = bool(getattr(_legacy_config, "USE_STREAMING", True))
    use_json_format: bool = bool(getattr(_legacy_config, "USE_JSON_FORMAT", False))
    enable_parallel_processing: bool = bool(getattr(_legacy_config, "ENABLE_PARALLEL_PROCESSING", True))
    max_parallel_workers: int = int(getattr(_legacy_config, "MAX_PARALLEL_WORKERS", 16))
    enable_parallel_retrieval: bool = bool(getattr(_legacy_config, "ENABLE_PARALLEL_RETRIEVAL", True))
    max_retrieval_workers: int = int(getattr(_legacy_config, "MAX_RETRIEVAL_WORKERS", 8))
    enable_planning: bool = bool(getattr(_legacy_config, "ENABLE_PLANNING", True))
    enable_reflection: bool = bool(getattr(_legacy_config, "ENABLE_REFLECTION", True))
    max_reflection_rounds: int = int(getattr(_legacy_config, "MAX_REFLECTION_ROUNDS", 2))


_CONFIG = SimpleMemConfig()


def _apply_to_legacy(cfg: SimpleMemConfig) -> None:
    _legacy_config.OPENAI_API_KEY = cfg.openai_api_key
    _legacy_config.OPENAI_BASE_URL = cfg.openai_base_url
    _legacy_config.LLM_MODEL = cfg.llm_model
    _legacy_config.LLM_TEMPERATURE = cfg.llm_temperature
    _legacy_config.LLM_TOP_P = cfg.llm_top_p if cfg.llm_top_p is not None else cfg.top_p
    _legacy_config.EMBEDDING_BASE_URL = cfg.embedding_base_url
    _legacy_config.EMBEDDING_API_KEY = cfg.embedding_api_key
    _legacy_config.EMBEDDING_MODEL = cfg.embedding_model
    _legacy_config.EMBEDDING_DIMENSION = cfg.embedding_dimension
    _legacy_config.EMBEDDING_CONTEXT_LENGTH = cfg.embedding_context_length
    _legacy_config.WINDOW_SIZE = cfg.window_size
    _legacy_config.OVERLAP_SIZE = cfg.overlap_size
    _legacy_config.SEMANTIC_TOP_K = cfg.semantic_top_k
    _legacy_config.KEYWORD_TOP_K = cfg.keyword_top_k
    _legacy_config.STRUCTURED_TOP_K = cfg.structured_top_k
    _legacy_config.LANCEDB_PATH = cfg.lancedb_path
    _legacy_config.MEMORY_TABLE_NAME = cfg.memory_table_name
    _legacy_config.ENABLE_THINKING = cfg.enable_thinking
    _legacy_config.USE_STREAMING = cfg.use_streaming
    _legacy_config.USE_JSON_FORMAT = cfg.use_json_format
    _legacy_config.ENABLE_PARALLEL_PROCESSING = cfg.enable_parallel_processing
    _legacy_config.MAX_PARALLEL_WORKERS = cfg.max_parallel_workers
    _legacy_config.ENABLE_PARALLEL_RETRIEVAL = cfg.enable_parallel_retrieval
    _legacy_config.MAX_RETRIEVAL_WORKERS = cfg.max_retrieval_workers
    _legacy_config.ENABLE_PLANNING = cfg.enable_planning
    _legacy_config.ENABLE_REFLECTION = cfg.enable_reflection
    _legacy_config.MAX_REFLECTION_ROUNDS = cfg.max_reflection_rounds


def get_config() -> SimpleMemConfig:
    return _CONFIG


def set_config(cfg: SimpleMemConfig) -> None:
    global _CONFIG
    _CONFIG = cfg
    _apply_to_legacy(cfg)
