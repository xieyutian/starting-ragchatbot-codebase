import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- Use the course outline tool for questions about course structure, curriculum, or lesson lists
- **Multi-step reasoning**: You may use tools sequentially when needed (up to 2 tool calls)
  - Example: First get a course outline to find a lesson title, then search that lesson's content
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Course outline questions**: Use the course outline tool to retrieve complete course information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                     conversation_history: Optional[str] = None,
                     tools: Optional[List] = None,
                     tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling up to max_tool_rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize messages
        messages = [{"role": "user", "content": query}]

        # Initial API call with tools
        api_params = self._build_api_params(messages, system_content, tools)
        response = self.client.messages.create(**api_params)

        # Sequential tool calling loop
        round_count = 0

        while response.stop_reason == "tool_use" and round_count < self.max_tool_rounds and tool_manager:
            # Execute tools and get results
            tool_results = self._execute_tools(response, tool_manager)

            # Update message history
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            # Next API call - include tools only if more rounds allowed
            include_tools = (round_count + 1) < self.max_tool_rounds
            api_params = self._build_api_params(
                messages, system_content,
                tools if include_tools else None
            )
            response = self.client.messages.create(**api_params)
            round_count += 1

        return response.content[0].text

    def _build_api_params(self, messages: List, system_content: str, tools: Optional[List] = None) -> Dict:
        """
        Build API parameters dictionary.

        Args:
            messages: Conversation messages
            system_content: System prompt content
            tools: Optional tools to include

        Returns:
            API parameters dictionary
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        return api_params

    def _execute_tools(self, response, tool_manager) -> List[Dict]:
        """
        Execute all tool_use blocks in the response.

        Args:
            response: API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            List of tool_result blocks
        """
        tool_results = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                except Exception as e:
                    result = f"Error executing {content_block.name}: {str(e)}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": result
                })

        return tool_results