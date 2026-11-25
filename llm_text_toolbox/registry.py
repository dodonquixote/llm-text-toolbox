"""Core tool registry for llm-text-toolbox.

A Tool is a plain Python function plus a JSON-schema description, so the
same tool can be exposed to an LLM as an OpenAI-style function definition,
or served over MCP (see server.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    tags: list[str] = field(default_factory=list)

    def function_definition(self) -> dict[str, Any]:
        """OpenAI-style function/tool definition for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, description: str, parameters: dict[str, Any],
             tags: list[str] | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of register()."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(Tool(name=name, description=description,
                               parameters=parameters, func=func, tags=tags or []))
            return func
        return decorator

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}. Available: {sorted(self._tools)}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def function_definitions(self) -> list[dict[str, Any]]:
        return [self._tools[n].function_definition() for n in self.names()]

    def call(self, name: str, arguments: dict[str, Any] | str) -> Any:
        """Call a tool by name. `arguments` may be a dict or a JSON string,
        which is how most LLM frameworks return tool-call arguments."""
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        tool = self.get(name)
        return tool.func(**arguments)


# Global registry that the bundled tools register themselves into.
registry = ToolRegistry()
