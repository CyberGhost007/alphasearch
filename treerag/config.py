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

    # Adaptive MCTS
    adaptive: bool = True
    min_iterations: int = 8
    max_iterations: int = 50
    meta_min_iterations: int = 5
    meta_max_iterations: int = 30

    # Convergence detection
    convergence_window: int = 4
    convergence_variance_threshold: float = 0.01
    top_k_stable_rounds: int = 3

    # Branch pruning
    pruning_enabled: bool = True
    pruning_min_visits: int = 3
    pruning_reward_threshold: float = 0.25

    # Adaptive exploration (C decay)
    adaptive_exploration: bool = True
    exploration_start: float = 2.0
    exploration_end: float = 0.5
    exploration_decay: str = "linear"

    # Batch simulation (parallel LLM calls per iteration)
    simulation_batch_size: int = 4


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
                adaptive=os.getenv("MCTS_ADAPTIVE", "true").lower() == "true",
                min_iterations=int(os.getenv("MCTS_MIN_ITERATIONS", "8")),
                max_iterations=int(os.getenv("MCTS_MAX_ITERATIONS", "50")),
                meta_min_iterations=int(os.getenv("MCTS_META_MIN_ITERATIONS", "5")),
                meta_max_iterations=int(os.getenv("MCTS_META_MAX_ITERATIONS", "30")),
                convergence_window=int(os.getenv("MCTS_CONVERGENCE_WINDOW", "4")),
                convergence_variance_threshold=float(os.getenv("MCTS_CONVERGENCE_VARIANCE_THRESHOLD", "0.01")),
                top_k_stable_rounds=int(os.getenv("MCTS_TOP_K_STABLE_ROUNDS", "3")),
                pruning_enabled=os.getenv("MCTS_PRUNING", "true").lower() == "true",
                pruning_min_visits=int(os.getenv("MCTS_PRUNING_MIN_VISITS", "3")),
                pruning_reward_threshold=float(os.getenv("MCTS_PRUNING_REWARD_THRESHOLD", "0.25")),
                adaptive_exploration=os.getenv("MCTS_ADAPTIVE_EXPLORATION", "true").lower() == "true",
                exploration_start=float(os.getenv("MCTS_EXPLORATION_START", "2.0")),
                exploration_end=float(os.getenv("MCTS_EXPLORATION_END", "0.5")),
                exploration_decay=os.getenv("MCTS_EXPLORATION_DECAY", "linear"),
                simulation_batch_size=int(os.getenv("MCTS_SIMULATION_BATCH_SIZE", "4")),
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
