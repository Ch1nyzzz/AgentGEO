from typing import Callable, Type, List
from pydantic import BaseModel
from langchain_core.tools import StructuredTool, BaseTool

class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, func: Callable, args_schema: Type[BaseModel] = None, name: str = None, description: str = None) -> BaseTool:
        """Register a function as a tool with optional Pydantic schema."""
        
        registered_tool = StructuredTool.from_function(
            func=func,
            name=name,
            description=description,
            args_schema=args_schema
        )
        
        self.tools[registered_tool.name] = registered_tool
        return registered_tool

    def get_tool(self, name: str):
        return self.tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        return list(self.tools.values())

registry = ToolRegistry()