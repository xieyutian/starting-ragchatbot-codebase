"""
Shared fixtures for RAG system tests.
"""

import os

# Add parent directory to path for imports
import sys
import tempfile
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


@dataclass
class TestConfig:
    """Test configuration with correct MAX_RESULTS"""

    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5  # Correct value for tests
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = ""  # Will be set by fixture


@pytest.fixture
def test_config():
    """Create test configuration with temporary ChromaDB path"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = TestConfig(CHROMA_PATH=temp_dir)
        yield config


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore for unit tests"""
    mock_store = MagicMock(spec=VectorStore)

    # Default search results
    mock_search_results = SearchResults(
        documents=["Test content about MCP protocol"],
        metadata=[
            {
                "course_title": "Introduction to MCP",
                "lesson_number": 1,
                "chunk_index": 0,
            }
        ],
        distances=[0.5],
    )
    mock_store.search.return_value = mock_search_results
    mock_store.get_lesson_link.return_value = "https://example.com/lesson/1"

    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client"""
    mock_client = MagicMock()

    # Mock response without tool use
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(text="This is a test response")]

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def sample_course():
    """Create a sample Course object for testing"""
    return Course(
        title="Introduction to MCP",
        instructor="Test Instructor",
        course_link="https://example.com/course",
        lessons=[
            Lesson(
                lesson_number=1,
                title="What is MCP?",
                content="MCP stands for Model Context Protocol...",
                lesson_link="https://example.com/lesson/1",
            ),
            Lesson(
                lesson_number=2,
                title="MCP Architecture",
                content="The MCP architecture consists of...",
                lesson_link="https://example.com/lesson/2",
            ),
        ],
    )


@pytest.fixture
def sample_chunks():
    """Create sample CourseChunk objects for testing"""
    return [
        CourseChunk(
            course_title="Introduction to MCP",
            lesson_number=1,
            chunk_index=0,
            content="MCP stands for Model Context Protocol. It is a protocol for AI assistants.",
        ),
        CourseChunk(
            course_title="Introduction to MCP",
            lesson_number=1,
            chunk_index=1,
            content="The protocol enables standardized communication between AI models.",
        ),
        CourseChunk(
            course_title="Introduction to MCP",
            lesson_number=2,
            chunk_index=0,
            content="The MCP architecture consists of clients, servers, and hosts.",
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Create sample SearchResults for testing"""
    return SearchResults(
        documents=[
            "MCP stands for Model Context Protocol. It is a protocol for AI assistants.",
            "The protocol enables standardized communication between AI models.",
        ],
        metadata=[
            {
                "course_title": "Introduction to MCP",
                "lesson_number": 1,
                "chunk_index": 0,
            },
            {
                "course_title": "Introduction to MCP",
                "lesson_number": 1,
                "chunk_index": 1,
            },
        ],
        distances=[0.3, 0.4],
    )


@pytest.fixture
def empty_search_results():
    """Create empty SearchResults for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Create SearchResults with error for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="No course found matching 'NonExistent'",
    )
