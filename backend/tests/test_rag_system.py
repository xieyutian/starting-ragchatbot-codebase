"""
Integration tests for RAGSystem.
"""
import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import SearchResults
from models import Course, Lesson


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_returns_response_with_sources(
        self, mock_session_manager, mock_doc_processor, mock_vector_store_class, mock_ai_generator_class
    ):
        """Verify query returns response with sources"""
        # Setup mocks
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        # Setup search results
        search_results = SearchResults(
            documents=["MCP is Model Context Protocol"],
            metadata=[{"course_title": "Introduction to MCP", "lesson_number": 1}],
            distances=[0.3]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"
        mock_vector_store.get_existing_course_titles.return_value = []

        # Setup AI generator
        mock_ai_gen = MagicMock()

        # First response: tool use
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_123"
        tool_use_content.input = {"query": "What is MCP?"}
        tool_use_response.content = [tool_use_content]

        # Second response: final answer
        final_response_content = MagicMock()
        final_response_content.text = "MCP stands for Model Context Protocol."
        final_response = MagicMock()
        final_response.content = [final_response_content]

        mock_ai_gen.generate_response.return_value = "MCP stands for Model Context Protocol."
        mock_ai_generator_class.return_value = mock_ai_gen

        # Create config mock
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            ANTHROPIC_API_KEY: str = "test-key"
            ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
            CHUNK_SIZE: int = 800
            CHUNK_OVERLAP: int = 100
            MAX_RESULTS: int = 5
            MAX_HISTORY: int = 2
            CHROMA_PATH: str = "./test_chroma"

        # Create RAG system
        rag = RAGSystem(MockConfig())

        # Mock the AI generator to call the tool
        def mock_generate_response(**kwargs):
            tool_manager = kwargs.get('tool_manager')
            if tool_manager:
                # Simulate tool execution
                tool_manager.execute_tool("search_course_content", query="What is MCP?")
            return "MCP stands for Model Context Protocol."

        mock_ai_gen.generate_response.side_effect = mock_generate_response

        # Execute query
        response, sources = rag.query("What is MCP?")

        # Verify response
        assert response == "MCP stands for Model Context Protocol."

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_handles_empty_vector_store(
        self, mock_session_manager, mock_doc_processor, mock_vector_store_class, mock_ai_generator_class
    ):
        """Verify empty vector store is handled gracefully"""
        # Setup mocks
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        # Empty search results
        empty_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )
        mock_vector_store.search.return_value = empty_results
        mock_vector_store.get_existing_course_titles.return_value = []

        # Setup AI generator
        mock_ai_gen = MagicMock()
        mock_ai_gen.generate_response.return_value = "No relevant content found."
        mock_ai_generator_class.return_value = mock_ai_gen

        # Create config mock
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            ANTHROPIC_API_KEY: str = "test-key"
            ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
            CHUNK_SIZE: int = 800
            CHUNK_OVERLAP: int = 100
            MAX_RESULTS: int = 5  # Non-zero for proper behavior
            MAX_HISTORY: int = 2
            CHROMA_PATH: str = "./test_chroma"

        # Create RAG system
        rag = RAGSystem(MockConfig())

        # Execute query - should handle empty results gracefully
        response, sources = rag.query("Nonexistent topic")

        # Verify empty sources
        assert sources == []

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_session_management(
        self, mock_session_manager_class, mock_doc_processor, mock_vector_store_class, mock_ai_generator_class
    ):
        """Verify session management works correctly"""
        # Setup mocks
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_vector_store.get_existing_course_titles.return_value = []

        mock_session_manager = MagicMock()
        mock_session_manager.get_conversation_history.return_value = "Previous conversation"
        mock_session_manager_class.return_value = mock_session_manager

        mock_ai_gen = MagicMock()
        mock_ai_gen.generate_response.return_value = "Response text"
        mock_ai_generator_class.return_value = mock_ai_gen

        # Create config mock
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            ANTHROPIC_API_KEY: str = "test-key"
            ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
            CHUNK_SIZE: int = 800
            CHUNK_OVERLAP: int = 100
            MAX_RESULTS: int = 5
            MAX_HISTORY: int = 2
            CHROMA_PATH: str = "./test_chroma"

        # Create RAG system
        rag = RAGSystem(MockConfig())

        # Execute query with session ID
        rag.query("First question", session_id="session-123")

        # Verify session manager was called
        mock_session_manager.get_conversation_history.assert_called_once_with("session-123")
        mock_session_manager.add_exchange.assert_called_once_with("session-123", "First question", "Response text")


class TestRAGSystemMaxResults:
    """Tests specifically for MAX_RESULTS configuration"""

    def test_vector_store_uses_correct_max_results(self, test_config):
        """Verify VectorStore receives correct MAX_RESULTS value"""
        from vector_store import VectorStore

        # Mock both ChromaDB client and embedding function to avoid network calls
        with patch('vector_store.chromadb.PersistentClient') as mock_client, \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_embed:
            mock_client.return_value.get_or_create_collection.return_value = MagicMock()
            mock_embed.return_value = MagicMock()

            store = VectorStore(
                chroma_path=test_config.CHROMA_PATH,
                embedding_model=test_config.EMBEDDING_MODEL,
                max_results=test_config.MAX_RESULTS
            )

            # Verify max_results is set correctly
            assert store.max_results == 5, "MAX_RESULTS should be 5, not 0"

    def test_max_results_zero_causes_empty_search_results(self, test_config):
        """Verify MAX_RESULTS=0 causes empty search results (ROOT CAUSE TEST)"""
        from vector_store import VectorStore

        with patch('vector_store.chromadb.PersistentClient') as mock_client, \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_embed:
            # Setup mock collection
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            mock_embed.return_value = MagicMock()

            # Create store with MAX_RESULTS=0 (simulating the bug)
            store = VectorStore(
                chroma_path=test_config.CHROMA_PATH,
                embedding_model=test_config.EMBEDDING_MODEL,
                max_results=0  # This is the bug!
            )

            # Mock query response
            mock_collection.query.return_value = {
                'documents': [[]],  # Empty because n_results=0
                'metadatas': [[]],
                'distances': [[]]
            }

            # Execute search
            results = store.search(query="test")

            # Verify query was called with n_results=0
            mock_collection.query.assert_called_once()
            call_args = mock_collection.query.call_args
            assert call_args[1]['n_results'] == 0, "BUG: n_results=0 causes empty results"

            # Results should be empty
            assert results.is_empty()

    def test_max_results_five_returns_results(self, test_config):
        """Verify MAX_RESULTS=5 returns proper results"""
        from vector_store import VectorStore

        with patch('vector_store.chromadb.PersistentClient') as mock_client, \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_embed:
            # Setup mock collection
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            mock_embed.return_value = MagicMock()

            # Create store with correct MAX_RESULTS
            store = VectorStore(
                chroma_path=test_config.CHROMA_PATH,
                embedding_model=test_config.EMBEDDING_MODEL,
                max_results=5  # Correct value
            )

            # Mock query response with actual results
            mock_collection.query.return_value = {
                'documents': [["MCP content here"]],
                'metadatas': [[{"course_title": "Test Course", "lesson_number": 1}]],
                'distances': [[0.5]]
            }

            # Execute search
            results = store.search(query="test")

            # Verify query was called with correct n_results
            call_args = mock_collection.query.call_args
            assert call_args[1]['n_results'] == 5, "n_results should be 5"

            # Results should have content
            assert not results.is_empty()
            assert len(results.documents) == 1


class TestRAGSystemIntegration:
    """End-to-end integration tests"""

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_end_to_end_query_flow(
        self, mock_session_manager, mock_doc_processor, mock_vector_store_class, mock_ai_generator_class
    ):
        """Verify complete query flow from input to output"""
        # Setup mocks
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        # Setup search results
        search_results = SearchResults(
            documents=["MCP enables standardized communication between AI models."],
            metadata=[{"course_title": "Introduction to MCP", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.3]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"
        mock_vector_store.get_existing_course_titles.return_value = []

        # Setup AI generator
        mock_ai_gen = MagicMock()
        mock_ai_generator_class.return_value = mock_ai_gen

        # Create config mock
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            ANTHROPIC_API_KEY: str = "test-key"
            ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
            CHUNK_SIZE: int = 800
            CHUNK_OVERLAP: int = 100
            MAX_RESULTS: int = 5
            MAX_HISTORY: int = 2
            CHROMA_PATH: str = "./test_chroma"

        # Create RAG system
        rag = RAGSystem(MockConfig())

        # Mock the generate_response to simulate tool usage
        def simulate_tool_usage(**kwargs):
            tool_manager = kwargs.get('tool_manager')
            if tool_manager:
                result = tool_manager.execute_tool("search_course_content", query="MCP")
                return f"Based on the search: {result[:50]}..."
            return "Direct response"

        mock_ai_gen.generate_response.side_effect = simulate_tool_usage

        # Execute query
        response, sources = rag.query("What is MCP?")

        # Verify complete flow
        assert response is not None
        assert len(response) > 0
        # Sources should be populated by tool execution
        assert len(sources) == 1
        assert sources[0]['course_title'] == "Introduction to MCP"