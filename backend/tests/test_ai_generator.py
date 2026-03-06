"""
Tests for AIGenerator tool calling functionality.
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGeneratorSequentialToolCalling:
    """Tests for sequential tool calling behavior"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_sequential_tool_calls_execute_both(
        self, mock_anthropic_class, mock_vector_store, sample_search_results
    ):
        """Verify two sequential tool calls are both executed"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First response: get_course_outline tool use
        outline_response = MagicMock()
        outline_response.stop_reason = "tool_use"
        outline_content = MagicMock()
        outline_content.type = "tool_use"
        outline_content.name = "get_course_outline"
        outline_content.id = "tool_001"
        outline_content.input = {"course_title": "MCP Course"}
        outline_response.content = [outline_content]

        # Second response: search_course_content tool use
        search_response = MagicMock()
        search_response.stop_reason = "tool_use"
        search_content = MagicMock()
        search_content.type = "tool_use"
        search_content.name = "search_course_content"
        search_content.id = "tool_002"
        search_content.input = {"query": "lesson 4 topic"}
        search_response.content = [search_content]

        # Third response: final answer
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [
            MagicMock(text="Lesson 4 discusses MCP architecture.")
        ]

        mock_client.messages.create.side_effect = [
            outline_response,
            search_response,
            final_response,
        ]

        # Setup tool manager with both tools
        from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))
        tool_manager.register_tool(CourseOutlineTool(mock_vector_store))
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "MCP Course",
                "lessons": [{"lesson_number": 4, "lesson_title": "MCP Architecture"}],
            }
        ]

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="What is lesson 4 about?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify three API calls were made (2 tool uses + final response)
        assert mock_client.messages.create.call_count == 3
        assert "MCP" in result or len(result) > 0

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_two_rounds_enforced(
        self, mock_anthropic_class, mock_vector_store, sample_search_results
    ):
        """Verify only 2 tool rounds are allowed, 3rd call forces text response"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use (round 0)
        response1 = MagicMock()
        response1.stop_reason = "tool_use"
        content1 = MagicMock()
        content1.type = "tool_use"
        content1.name = "search_course_content"
        content1.id = "tool_001"
        content1.input = {"query": "query1"}
        response1.content = [content1]

        # Second response: tool use again (round 1)
        response2 = MagicMock()
        response2.stop_reason = "tool_use"
        content2 = MagicMock()
        content2.type = "tool_use"
        content2.name = "search_course_content"
        content2.id = "tool_002"
        content2.input = {"query": "query2"}
        response2.content = [content2]

        # Third response: forced text response (tools removed, max rounds reached)
        response3 = MagicMock()
        response3.stop_reason = "end_turn"
        response3.content = [MagicMock(text="Based on my searches...")]

        mock_client.messages.create.side_effect = [response1, response2, response3]

        # Setup tool manager
        from search_tools import CourseSearchTool, ToolManager

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))
        mock_vector_store.search.return_value = sample_search_results

        # Run generator with max_tool_rounds=2
        generator = AIGenerator(
            api_key="test-key", model="claude-sonnet-4-20250514", max_tool_rounds=2
        )
        result = generator.generate_response(
            query="complex query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify 3 API calls: round 0 (with tools), round 1 (with tools), round 2 (no tools - forced text)
        assert mock_client.messages.create.call_count == 3

        # Verify third API call had no tools (forcing text response)
        third_call_args = mock_client.messages.create.call_args_list[2]
        assert (
            "tools" not in third_call_args[1] or third_call_args[1].get("tools") is None
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_single_tool_unchanged(
        self, mock_anthropic_class, mock_vector_store, sample_search_results
    ):
        """Verify single tool call behavior remains unchanged"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

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
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [
            MagicMock(text="MCP stands for Model Context Protocol.")
        ]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup tool manager
        from search_tools import CourseSearchTool, ToolManager

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))
        mock_vector_store.search.return_value = sample_search_results

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="What is MCP?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify two API calls (tool use + final response)
        assert mock_client.messages.create.call_count == 2
        assert "MCP" in result

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_error_recovery(self, mock_anthropic_class, mock_vector_store):
        """Verify tool execution errors are handled gracefully"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_err"
        tool_use_content.input = {"query": "test"}
        tool_use_response.content = [tool_use_content]

        # Final response after error
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [
            MagicMock(text="I encountered an error but here's what I know...")
        ]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup tool manager with tool that raises exception
        from search_tools import CourseSearchTool, ToolManager

        tool_manager = ToolManager()
        mock_vector_store.search.side_effect = Exception("Database connection failed")
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify error was passed as tool result and Claude recovered
        assert mock_client.messages.create.call_count == 2

        # Verify second call included error message in tool results
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        tool_result_content = messages[-1]["content"]
        assert any(
            "error" in str(block.get("content", "")).lower()
            for block in tool_result_content
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_empty_tool_result_handling(self, mock_anthropic_class, mock_vector_store):
        """Verify empty tool results are handled gracefully"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_empty"
        tool_use_content.input = {"query": "nonexistent"}
        tool_use_response.content = [tool_use_content]

        # Final response
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [MagicMock(text="No relevant content was found.")]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup tool manager with empty results
        from search_tools import CourseSearchTool, ToolManager
        from vector_store import SearchResults

        tool_manager = ToolManager()
        empty_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        mock_vector_store.search.return_value = empty_results
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="nonexistent topic",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify tool was executed and Claude handled empty results
        assert mock_client.messages.create.call_count == 2
        assert "not found" in result.lower() or "no" in result.lower()

    @patch("ai_generator.anthropic.Anthropic")
    def test_sources_accumulate_across_rounds(
        self, mock_anthropic_class, mock_vector_store, sample_search_results
    ):
        """Verify sources accumulate across multiple tool calls"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        response1 = MagicMock()
        response1.stop_reason = "tool_use"
        content1 = MagicMock()
        content1.type = "tool_use"
        content1.name = "search_course_content"
        content1.id = "tool_001"
        content1.input = {"query": "query1"}
        response1.content = [content1]

        # Second response: tool use again
        response2 = MagicMock()
        response2.stop_reason = "tool_use"
        content2 = MagicMock()
        content2.type = "tool_use"
        content2.name = "search_course_content"
        content2.id = "tool_002"
        content2.input = {"query": "query2"}
        response2.content = [content2]

        # Third response: final answer
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [MagicMock(text="Combined answer from both searches.")]

        mock_client.messages.create.side_effect = [response1, response2, final_response]

        # Setup tool manager
        from search_tools import CourseSearchTool, ToolManager

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        mock_vector_store.search.return_value = sample_search_results

        # Reset sources before test
        search_tool.last_sources = []

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="multi-step query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify sources accumulated from both tool calls
        sources = tool_manager.get_last_sources()
        # sample_search_results has 2 items, so 2 tool calls = 4 sources
        assert len(sources) == 4


class TestAIGeneratorToolUsage:
    """Tests for AIGenerator tool calling behavior"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generator_calls_search_tool_for_content_query(
        self, mock_anthropic_class, mock_vector_store, sample_search_results
    ):
        """Verify content queries trigger tool calls"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

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
        final_response = MagicMock()
        final_response.content = [
            MagicMock(text="MCP stands for Model Context Protocol.")
        ]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup tool manager
        from search_tools import CourseSearchTool, ToolManager

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))
        mock_vector_store.search.return_value = sample_search_results

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="What is MCP?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify tool was called
        mock_vector_store.search.assert_called()
        assert "MCP" in result or len(result) > 0

    @patch("ai_generator.anthropic.Anthropic")
    def test_generator_skips_tool_for_general_query(self, mock_anthropic_class):
        """Verify general queries don't trigger tool calls"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Response without tool use
        direct_response = MagicMock()
        direct_response.stop_reason = "end_turn"
        direct_response.content = [MagicMock(text="The capital of France is Paris.")]

        mock_client.messages.create.return_value = direct_response

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(query="What is the capital of France?")

        # Verify no tool use
        assert result == "The capital of France is Paris."
        assert mock_client.messages.create.call_count == 1

    @patch("ai_generator.anthropic.Anthropic")
    def test_generator_handles_tool_result_correctly(
        self, mock_anthropic_class, mock_vector_store, sample_search_results
    ):
        """Verify tool results are processed and passed back to model"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_456"
        tool_use_content.input = {"query": "MCP architecture"}
        tool_use_response.content = [tool_use_content]

        # Second response: final answer with tool result
        final_response = MagicMock()
        final_response.content = [
            MagicMock(text="The MCP architecture has three main components.")
        ]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup tool manager
        from search_tools import CourseSearchTool, ToolManager

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))
        mock_vector_store.search.return_value = sample_search_results

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="Tell me about MCP architecture",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify two API calls were made (tool use + final response)
        assert mock_client.messages.create.call_count == 2

        # Verify second call included tool results
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        # The last message should be tool results
        assert messages[-1]["role"] == "user"

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_flow(
        self, mock_anthropic_class, mock_vector_store, sample_search_results
    ):
        """Verify complete tool execution flow"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_789"
        tool_use_content.input = {"query": "test query", "course_name": "MCP"}
        tool_use_response.content = [tool_use_content]

        # Second response: final answer
        final_response = MagicMock()
        final_response.content = [MagicMock(text="Based on the course materials...")]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup tool manager
        from search_tools import CourseSearchTool, ToolManager

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))
        mock_vector_store.search.return_value = sample_search_results

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="MCP", lesson_number=None
        )

        # Verify sources were tracked
        sources = tool_manager.get_last_sources()
        assert len(sources) == 2


class TestAIGeneratorBasicFunctionality:
    """Tests for basic AIGenerator functionality"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_history(self, mock_anthropic_class):
        """Verify conversation history is included in the prompt"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Follow-up response")]

        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="Tell me more",
            conversation_history="User: What is MCP?\nAssistant: MCP is...",
        )

        # Verify history was included in system prompt
        call_args = mock_client.messages.create.call_args
        system_prompt = call_args[1]["system"]
        assert "Previous conversation:" in system_prompt
        assert "What is MCP?" in system_prompt

    @patch("ai_generator.anthropic.Anthropic")
    def test_base_params_are_used(self, mock_anthropic_class):
        """Verify base parameters are correctly set"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Response")]

        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.generate_response(query="test")

        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
