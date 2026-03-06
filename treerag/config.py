"""Configuration management for TreeRAG."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    indexing_model: str = "gpt-4o"
    search_model: str = "gpt-4o-mini"
    answer_model: str = "gpt-4o"
    api_key: str = ""
    max_retries: int = 3
    temperature: float = 0.0

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        # API key validated lazily in LLMClient — allows folder management without key


@dataclass
class MCTSConfig:
    iterations: int = 25
    meta_iterations: int = 15
    exploration_constant: float = 1.414
    confidence_threshold: float = 0.7
    top_k_results: int = 3
    top_k_documents: int = 3
    max_depth: int = 10
    parallel_phase2: bool = True


@dataclass
class IndexerConfig:
    batch_size: int = 15
    max_tokens_per_node: int = 20000
    image_dpi: int = 200
    overlap_pages: int = 2
    summary_max_words: int = 60
    keywords_per_node: int = 5
    max_pages: int = 5000              # Hard limit on document size


@dataclass
class FolderConfig:
    base_dir: str = ".treerag_data"
    cache_dir: str = "indices"
    auto_reindex: bool = True

    def get_folder_dir(self, folder_name: str) -> str:
        return os.path.join(self.base_dir, "folders", folder_name)

    def get_index_dir(self, folder_name: str) -> str:
        return os.path.join(self.get_folder_dir(folder_name), self.cache_dir)


@dataclass
class TreeRAGConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    indexer: IndexerConfig = field(default_factory=IndexerConfig)
    folder: FolderConfig = field(default_factory=FolderConfig)
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "TreeRAGConfig":
        return cls(
            model=ModelConfig(
                indexing_model=os.getenv("INDEXING_MODEL", "gpt-4o"),
                search_model=os.getenv("SEARCH_MODEL", "gpt-4o-mini"),
                answer_model=os.getenv("ANSWER_MODEL", "gpt-4o"),
            ),
            mcts=MCTSConfig(
                iterations=int(os.getenv("MCTS_ITERATIONS", "25")),
                meta_iterations=int(os.getenv("MCTS_META_ITERATIONS", "15")),
                exploration_constant=float(os.getenv("UCB_EXPLORATION_CONSTANT", "1.414")),
                confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
                top_k_documents=int(os.getenv("TOP_K_DOCUMENTS", "3")),
                parallel_phase2=os.getenv("PARALLEL_PHASE2", "true").lower() == "true",
            ),
            indexer=IndexerConfig(
                batch_size=int(os.getenv("BATCH_SIZE", "15")),
                max_pages=int(os.getenv("MAX_PAGES", "5000")),
            ),
            folder=FolderConfig(
                base_dir=os.getenv("TREERAG_DATA_DIR", ".treerag_data"),
                auto_reindex=os.getenv("AUTO_REINDEX", "true").lower() == "true",
            ),
        )
