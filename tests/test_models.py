"""Tests for data models."""

import json
import tempfile
from pathlib import Path

from treerag.models import TreeNode, DocumentIndex, SearchStats


class TestTreeNode:
    def test_create_node(self):
        node = TreeNode(node_id="0001", title="Test Section", summary="A test section",
                        start_page=0, end_page=5)
        assert node.node_id == "0001"
        assert node.title == "Test Section"
        assert node.start_page == 0
        assert node.end_page == 5

    def test_is_leaf(self):
        parent = TreeNode(node_id="0000", title="Root", summary="Root node",
                          start_page=0, end_page=10)
        assert parent.is_leaf is True

        child = TreeNode(node_id="0001", title="Child", summary="Child node",
                         start_page=0, end_page=5, parent=parent)
        parent.children.append(child)
        assert parent.is_leaf is False
        assert child.is_leaf is True

    def test_ucb1_unvisited(self):
        node = TreeNode(node_id="0001", title="Test", summary="Test",
                        start_page=0, end_page=5)
        # Unvisited node should return infinity
        assert node.ucb1(1.414) == float("inf")

    def test_ucb1_visited(self):
        node = TreeNode(node_id="0001", title="Test", summary="Test",
                        start_page=0, end_page=5)
        node.visit_count = 5
        node.total_reward = 3.0
        score = node.ucb1(1.414)
        assert score > 0
        assert score < float("inf")

    def test_average_reward(self):
        node = TreeNode(node_id="0001", title="Test", summary="Test",
                        start_page=0, end_page=5)
        assert node.average_reward == 0.0
        node.visit_count = 4
        node.total_reward = 2.0
        assert node.average_reward == 0.5

    def test_from_dict(self):
        data = {
            "node_id": "0000",
            "title": "Root",
            "summary": "Root summary",
            "keywords": ["test"],
            "start_page": 0,
            "end_page": 10,
            "level": 0,
            "children": [
                {
                    "node_id": "0001",
                    "title": "Child",
                    "summary": "Child summary",
                    "keywords": [],
                    "start_page": 0,
                    "end_page": 5,
                    "level": 1,
                    "children": [],
                }
            ],
        }
        node = TreeNode.from_dict(data)
        assert node.node_id == "0000"
        assert len(node.children) == 1
        assert node.children[0].title == "Child"

    def test_reset_mcts_state(self):
        node = TreeNode(node_id="0000", title="Root", summary="Root",
                        start_page=0, end_page=10)
        child = TreeNode(node_id="0001", title="Child", summary="Child",
                         start_page=0, end_page=5, parent=node)
        node.children.append(child)
        node.visit_count = 10
        node.total_reward = 5.0
        child.visit_count = 3
        child.total_reward = 1.5
        node.reset_mcts_state()
        assert node.visit_count == 0
        assert node.total_reward == 0.0
        assert child.visit_count == 0
        assert child.total_reward == 0.0

    def test_pruned_default(self):
        node = TreeNode(node_id="0001", title="Test", summary="Test",
                        start_page=0, end_page=5)
        assert node.pruned is False

    def test_pruned_resets(self):
        node = TreeNode(node_id="0000", title="Root", summary="Root",
                        start_page=0, end_page=10)
        child = TreeNode(node_id="0001", title="Child", summary="Child",
                         start_page=0, end_page=5, parent=node)
        node.children.append(child)
        node.pruned = True
        child.pruned = True
        node.reset_mcts_state()
        assert node.pruned is False
        assert child.pruned is False


class TestDocumentIndex:
    def test_save_and_load(self):
        root = TreeNode(node_id="0000", title="Doc", summary="Test doc",
                        start_page=0, end_page=5)
        doc = DocumentIndex(
            document_id="abc123", filename="test.pdf",
            total_pages=6, description="A test document", root=root,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            doc.save(tmp_path)
            loaded = DocumentIndex.load(tmp_path)
            assert loaded.document_id == "abc123"
            assert loaded.filename == "test.pdf"
            assert loaded.total_pages == 6
            assert loaded.root.title == "Doc"
        finally:
            tmp_path.unlink()


class TestSearchStats:
    def test_defaults(self):
        stats = SearchStats()
        assert stats.iterations_used == 0
        assert stats.converged is False
        assert stats.convergence_reason == ""
        assert stats.pruned_branches == 0
        assert stats.coverage_pct == 0.0

    def test_to_dict(self):
        stats = SearchStats(
            iterations_used=12, iterations_max=50,
            converged=True, convergence_reason="top_k_stable",
            convergence_iteration=11,
            total_nodes=47, visited_nodes=38,
            coverage_pct=80.85,
            pruned_branches=3,
            mean_reward=0.7234, reward_variance=0.0456,
        )
        d = stats.to_dict()
        assert d["iterations_used"] == 12
        assert d["converged"] is True
        assert d["convergence_reason"] == "top_k_stable"
        assert d["coverage_pct"] == 80.8  # rounded
        assert d["pruned_branches"] == 3
        assert d["mean_reward"] == 0.7234
        assert d["reward_variance"] == 0.0456
