"""
Tests for AIGenerator tool calling functionality.
"""
import pytest
import os
import sys
from unittest.mock import MagicMock, patch, ANY

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGeneratorToolUsage:
    """Tests for AIGenerator tool calling behavior"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_generator_calls_search_tool_for_content_query(self, mock_anthropic_class, mock_vector_store, sample_search_results):
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
        final_response.content = [MagicMock(text="MCP stands for Model Context Protocol.")]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup tool manager
        from search_tools import ToolManager, CourseSearchTool
        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))
        mock_vector_store.search.return_value = sample_search_results

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="What is MCP?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify tool was called
        mock_vector_store.search.assert_called()
        assert "MCP" in result or len(result) > 0

    @patch('ai_generator.anthropic.Anthropic')
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

    @patch('ai_generator.anthropic.Anthropic')
    def test_generator_handles_tool_result_correctly(self, mock_anthropic_class, mock_vector_store, sample_search_results):
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
        final_response.content = [MagicMock(text="The MCP architecture has three main components.")]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup tool manager
        from search_tools import ToolManager, CourseSearchTool
        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))
        mock_vector_store.search.return_value = sample_search_results

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="Tell me about MCP architecture",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify two API calls were made (tool use + final response)
        assert mock_client.messages.create.call_count == 2

        # Verify second call included tool results
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        # The last message should be tool results
        assert messages[-1]['role'] == 'user'

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_flow(self, mock_anthropic_class, mock_vector_store, sample_search_results):
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
        from search_tools import ToolManager, CourseSearchTool
        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))
        mock_vector_store.search.return_value = sample_search_results

        # Run generator
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="MCP",
            lesson_number=None
        )

        # Verify sources were tracked
        sources = tool_manager.get_last_sources()
        assert len(sources) == 2


class TestAIGeneratorBasicFunctionality:
    """Tests for basic AIGenerator functionality"""

    @patch('ai_generator.anthropic.Anthropic')
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
            conversation_history="User: What is MCP?\nAssistant: MCP is..."
        )

        # Verify history was included in system prompt
        call_args = mock_client.messages.create.call_args
        system_prompt = call_args[1]['system']
        assert "Previous conversation:" in system_prompt
        assert "What is MCP?" in system_prompt

    @patch('ai_generator.anthropic.Anthropic')
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
        assert call_args['model'] == "claude-sonnet-4-20250514"
        assert call_args['temperature'] == 0
        assert call_args['max_tokens'] == 800