"""llm-text-toolbox: text utilities for LLM and AI-agent workflows.

Importing the package registers all bundled tools into `registry`.
"""
from .registry import Tool, ToolRegistry, registry

# Import tool modules so their @registry.tool decorators run.
from . import tools as _tools  # noqa: F401

__all__ = ["Tool", "ToolRegistry", "registry"]
__version__ = "0.1.0"
