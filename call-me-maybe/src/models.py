from pydantic import BaseModel, field_validator, ValidationError
from typing import Dict, List


class PromptInput(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be empty")
        return value


ALLOWED_TYPES = {"string", "number"}


class ParameterDefinition(BaseModel):
    type: str

    model_config = {
        "extra": "forbid"
    }

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value not in ALLOWED_TYPES:
            raise ValueError(f"Invalid type '{value}'. Allowed types: {ALLOWED_TYPES}")
        return value


class ReturnDefinition(BaseModel):
    type: str

    model_config = {
        "extra": "forbid"
    }

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value not in ALLOWED_TYPES:
            raise ValueError(f"Invalid return type '{value}'. Allowed types: {ALLOWED_TYPES}")
        return value


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: ReturnDefinition

    model_config = {
        "extra": "forbid"
    }

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Function name cannot be empty")
        return value

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Description cannot be empty")
        return value