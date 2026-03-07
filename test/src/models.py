from pydantic import BaseModel, field_validator
from typing import Any, Dict


# Allowed parameter types in function definitions
ALLOWED_TYPES = {"string", "number", "boolean"}


class PromptInput(BaseModel):
    """Model representing a single user prompt.

    Attributes:
        prompt: The natural language prompt string.
    """

    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, value: str) -> str:
        """Validate that prompt is not empty or whitespace-only."""
        if not value.strip():
            raise ValueError("Prompt cannot be empty")
        return value


class ParameterDefinition(BaseModel):
    """Model representing a function parameter type definition.

    Attributes:
        type: The type of the parameter (string, number, boolean).
    """

    type: str

    model_config = {"extra": "forbid"}

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Validate parameter type is one of the allowed types."""
        if value not in ALLOWED_TYPES:
            raise ValueError(
                f"Invalid type '{value}'. "
                f"Allowed types: {ALLOWED_TYPES}"
            )
        return value


class ReturnDefinition(BaseModel):
    """Model representing a function return type definition.

    Attributes:
        type: The return type of the function (string, number, boolean).
    """

    type: str

    model_config = {"extra": "forbid"}

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Validate return type is one of the allowed types."""
        if value not in ALLOWED_TYPES:
            raise ValueError(
                f"Invalid return type '{value}'. "
                f"Allowed types: {ALLOWED_TYPES}"
            )
        return value


class FunctionDefinition(BaseModel):
    """Model representing a function definition with its parameters.

    Attributes:
        name: The function name (e.g. 'fn_add_numbers').
        description: Human-readable description of the function.
        parameters: Dict mapping parameter names to their type definitions.
        returns: The return type definition.
    """

    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: ReturnDefinition

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        """Validate that function name is not empty."""
        if not value.strip():
            raise ValueError("Function name cannot be empty")
        return value

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, value: str) -> str:
        """Validate that function description is not empty."""
        if not value.strip():
            raise ValueError("Description cannot be empty")
        return value


class FunctionCallResult(BaseModel):
    """Model representing the output of a function call extraction.

    Attributes:
        prompt: The original natural language prompt.
        name: The name of the function to call.
        parameters: Dict of argument names to their values.
    """

    prompt: str
    name: str
    parameters: Dict[str, Any]

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        """Validate that function name is not empty."""
        if not value.strip():
            raise ValueError("Function name cannot be empty")
        return value
