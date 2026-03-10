"""Tests for configuration management."""

import os
from unittest.mock import patch

from treerag.config import TreeRAGConfig, ModelConfig, MCTSConfig, IndexerConfig, FolderConfig


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig()
        assert config.indexing_model == "gpt-4o"
        assert config.search_model == "gpt-4o-mini"
        assert config.answer_model == "gpt-4o"
        assert config.max_retries == 3
        assert config.temperature == 0.0

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            config = ModelConfig()
            assert config.api_key == "sk-test-key"


class TestMCTSConfig:
    def test_defaults(self):
        config = MCTSConfig()
        assert config.iterations == 25
        assert config.meta_iterations == 15
        assert config.exploration_constant == 1.414
        assert config.confidence_threshold == 0.7
        assert config.top_k_results == 3
        assert config.top_k_documents == 3
        assert config.parallel_phase2 is True

    def test_adaptive_defaults(self):
        config = MCTSConfig()
        assert config.adaptive is True
        assert config.min_iterations == 8
        assert config.max_iterations == 50
        assert config.meta_min_iterations == 5
        assert config.meta_max_iterations == 30
        assert config.convergence_window == 4
        assert config.convergence_variance_threshold == 0.01
        assert config.top_k_stable_rounds == 3
        assert config.pruning_enabled is True
        assert config.pruning_min_visits == 3
        assert config.pruning_reward_threshold == 0.25
        assert config.adaptive_exploration is True
        assert config.exploration_start == 2.0
        assert config.exploration_end == 0.5
        assert config.exploration_decay == "linear"
        assert config.simulation_batch_size == 4

    def test_adaptive_disabled(self):
        config = MCTSConfig(adaptive=False)
        assert config.adaptive is False
        # Other adaptive defaults still set
        assert config.min_iterations == 8
        assert config.max_iterations == 50


class TestIndexerConfig:
    def test_defaults(self):
        config = IndexerConfig()
        assert config.batch_size == 15
        assert config.max_pages == 5000
        assert config.image_dpi == 200


class TestFolderConfig:
    def test_defaults(self):
        config = FolderConfig()
        assert config.base_dir == ".treerag_data"
        assert config.auto_reindex is True

    def test_get_folder_dir(self):
        config = FolderConfig()
        assert config.get_folder_dir("test") == os.path.join(".treerag_data", "folders", "test")


class TestTreeRAGConfig:
    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            config = TreeRAGConfig.from_env()
            assert config.mcts.iterations == 25
            assert config.indexer.batch_size == 15
            assert config.folder.base_dir == ".treerag_data"

    def test_from_env_custom(self):
        env = {
            "MCTS_ITERATIONS": "40",
            "BATCH_SIZE": "10",
            "TREERAG_DATA_DIR": "/tmp/test_data",
        }
        with patch.dict(os.environ, env):
            config = TreeRAGConfig.from_env()
            assert config.mcts.iterations == 40
            assert config.indexer.batch_size == 10
            assert config.folder.base_dir == "/tmp/test_data"

    def test_from_env_adaptive(self):
        env = {
            "MCTS_ADAPTIVE": "false",
            "MCTS_MIN_ITERATIONS": "10",
            "MCTS_MAX_ITERATIONS": "60",
            "MCTS_PRUNING": "false",
            "MCTS_EXPLORATION_START": "1.8",
        }
        with patch.dict(os.environ, env):
            config = TreeRAGConfig.from_env()
            assert config.mcts.adaptive is False
            assert config.mcts.min_iterations == 10
            assert config.mcts.max_iterations == 60
            assert config.mcts.pruning_enabled is False
            assert config.mcts.exploration_start == 1.8
            assert config.mcts.simulation_batch_size == 4  # default unchanged
