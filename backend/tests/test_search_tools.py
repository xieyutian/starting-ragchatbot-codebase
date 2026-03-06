"""
Tests for CourseSearchTool and CourseOutlineTool.
"""
import pytest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    def test_execute_returns_results_when_data_exists(self, mock_vector_store, sample_search_results):
        """Verify normal search returns formatted results"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?")

        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name=None,
            lesson_number=None
        )

        # Verify result contains course info
        assert "Introduction to MCP" in result
        assert "Lesson 1" in result

    def test_execute_returns_empty_when_max_results_zero(self, mock_vector_store):
        """Verify MAX_RESULTS=0 returns empty results (ROOT CAUSE TEST)"""
        # Configure mock to return empty results (simulating MAX_RESULTS=0)
        empty_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )
        mock_vector_store.search.return_value = empty_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?")

        # Should return "No relevant content found" message
        assert "No relevant content found" in result

    def test_execute_handles_course_name_filter(self, mock_vector_store, sample_search_results):
        """Verify course name filter is passed to search"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?", course_name="Introduction")

        # Verify course_name was passed to search
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name="Introduction",
            lesson_number=None
        )

    def test_execute_handles_lesson_number_filter(self, mock_vector_store, sample_search_results):
        """Verify lesson number filter is passed to search"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?", lesson_number=2)

        # Verify lesson_number was passed to search
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name=None,
            lesson_number=2
        )

    def test_execute_handles_nonexistent_course(self, mock_vector_store):
        """Verify nonexistent course returns appropriate error message"""
        error_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="No course found matching 'NonExistent'"
        )
        mock_vector_store.search.return_value = error_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is X?", course_name="NonExistent")

        # Should return the error message
        assert "No course found matching 'NonExistent'" in result

    def test_format_results_includes_metadata(self, mock_vector_store, sample_search_results):
        """Verify formatted results include course title and lesson number"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?")

        # Verify format includes context header
        assert "[Introduction to MCP - Lesson 1]" in result

        # Verify sources were tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]['course_title'] == "Introduction to MCP"
        assert tool.last_sources[0]['lesson_number'] == 1

    def test_format_results_includes_lesson_link(self, mock_vector_store, sample_search_results):
        """Verify lesson links are included in sources when available"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="What is MCP?")

        # Verify lesson link was retrieved and added to sources
        mock_vector_store.get_lesson_link.assert_called()
        assert tool.last_sources[0].get('lesson_link') == "https://example.com/lesson/1"


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool"""

    def test_execute_returns_course_outline(self, mock_vector_store):
        """Verify outline tool returns formatted course information"""
        mock_vector_store.get_all_courses_metadata.return_value = [{
            'title': 'Introduction to MCP',
            'instructor': 'Test Instructor',
            'course_link': 'https://example.com/course',
            'lesson_count': 2,
            'lessons': [
                {'lesson_number': 1, 'lesson_title': 'What is MCP?', 'lesson_link': 'https://example.com/lesson/1'},
                {'lesson_number': 2, 'lesson_title': 'MCP Architecture', 'lesson_link': 'https://example.com/lesson/2'}
            ]
        }]

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="Introduction to MCP")

        # Verify outline contains expected information
        assert "Course Title: Introduction to MCP" in result
        assert "Course Instructor: Test Instructor" in result
        assert "Lesson 1: What is MCP?" in result
        assert "Lesson 2: MCP Architecture" in result

    def test_execute_handles_nonexistent_course(self, mock_vector_store):
        """Verify outline tool handles nonexistent course"""
        mock_vector_store.get_all_courses_metadata.return_value = []

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="NonExistent Course")

        assert "not found" in result


class TestToolManager:
    """Tests for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Verify tool registration works correctly"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_execute_tool(self, mock_vector_store, sample_search_results):
        """Verify tool execution through manager"""
        mock_vector_store.search.return_value = sample_search_results

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert "[Introduction to MCP" in result

    def test_get_tool_definitions(self, mock_vector_store):
        """Verify tool definitions are returned correctly"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        tool_names = [d['name'] for d in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Verify sources are tracked through manager"""
        mock_vector_store.search.return_value = sample_search_results

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search
        manager.execute_tool("search_course_content", query="test")

        # Get sources
        sources = manager.get_last_sources()

        assert len(sources) == 2
        assert sources[0]['course_title'] == "Introduction to MCP"

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Verify sources are reset correctly"""
        mock_vector_store.search.return_value = sample_search_results

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search
        manager.execute_tool("search_course_content", query="test")
        assert len(manager.get_last_sources()) == 2

        # Reset sources
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0