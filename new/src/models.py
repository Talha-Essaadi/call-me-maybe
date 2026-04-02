from pydantic import BaseModel, field_validator
from typing import Dict


class PromptInput(BaseModel):
    """Represent a single user prompt.

    Attributes
    ----------
    prompt : str
        Natural language request to convert into a function call.
    """

    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, value: str) -> str:
        """Validate that the prompt is not empty.

        Parameters
        ----------
        value : str
            Prompt text to validate.

        Returns
        -------
        str
            Original prompt when valid.

        Raises
        ------
        ValueError
            If the prompt is empty or whitespace only.
        """
        if not value.strip():
            raise ValueError("Prompt cannot be empty")
        return value


ALLOWED_TYPES = {"string", "number", "boolean", "integer"}


class ParameterDefinition(BaseModel):
    """Describe one function parameter type.

    Attributes
    ----------
    type : str
        Parameter type constrained to project-supported primitive types.
    """

    type: str

    model_config = {
        "extra": "forbid"
    }

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Validate that a parameter type is allowed.

        Parameters
        ----------
        value : str
            Declared parameter type.

        Returns
        -------
        str
            Original type when valid.

        Raises
        ------
        ValueError
            If the type is not in ``ALLOWED_TYPES``.
        """
        if value not in ALLOWED_TYPES:
            raise ValueError(
                f"Invalid type '{value}'. Allowed types: {ALLOWED_TYPES}")
        return value


class ReturnDefinition(BaseModel):
    """Describe one function return type.

    Attributes
    ----------
    type : str
        Return type constrained to project-supported primitive types.
    """

    type: str

    model_config = {
        "extra": "forbid"
    }

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Validate that a return type is allowed.

        Parameters
        ----------
        value : str
            Declared return type.

        Returns
        -------
        str
            Original type when valid.

        Raises
        ------
        ValueError
            If the type is not in ``ALLOWED_TYPES``.
        """
        if value not in ALLOWED_TYPES:
            raise ValueError(
                "Invalid return type "
                f"'{value}'. Allowed types: {ALLOWED_TYPES}"
            )
        return value


class FunctionDefinition(BaseModel):
    """Represent one callable function specification.

    Attributes
    ----------
    name : str
        Unique function identifier.
    description : str
        Human-readable function description.
    parameters : dict[str, ParameterDefinition]
        Typed parameter definitions by name.
    returns : ReturnDefinition
        Return value type definition.
    """

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
        """Validate that the function name is not empty.

        Parameters
        ----------
        value : str
            Function name to validate.

        Returns
        -------
        str
            Original name when valid.

        Raises
        ------
        ValueError
            If the function name is empty or whitespace only.
        """
        if not value.strip():
            raise ValueError("Function name cannot be empty")
        return value

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, value: str) -> str:
        """Validate that the function description is not empty.

        Parameters
        ----------
        value : str
            Description text to validate.

        Returns
        -------
        str
            Original description when valid.

        Raises
        ------
        ValueError
            If the description is empty or whitespace only.
        """
        if not value.strip():
            raise ValueError("Description cannot be empty")
        return value
