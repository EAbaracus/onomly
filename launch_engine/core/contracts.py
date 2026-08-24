"""Core contracts for Launch Engine."""

from pydantic import BaseModel
from typing import Protocol


class BaseModuleInput(BaseModel):
    """Base input for all modules."""


class BaseModuleOutput(BaseModel):
    """Base output for all modules."""


class LaunchModule(Protocol):
    """Protocol for launch modules."""

    name: str

    async def run(self, input_data: BaseModuleInput) -> BaseModuleOutput:
        """Run the module with the given input data."""
        ...
