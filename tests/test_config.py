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
