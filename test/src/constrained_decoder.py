import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .models import FunctionDefinition


class TokenVocabulary:
    """Manages the mapping between token IDs and their string forms.

    Args:
        vocabulary: Dict mapping token strings to token IDs.
    """

    def __init__(self, vocabulary: Dict[str, int]) -> None:
        """Initialize vocabulary mappings.

        Args:
            vocabulary: Token string -> token ID mapping.
        """
        self.str_to_id: Dict[str, int] = vocabulary
        self.id_to_str: Dict[int, str] = {
            v: k for k, v in vocabulary.items()
        }
        self.vocab_size: int = (
            max(vocabulary.values()) + 1 if vocabulary else 0
        )

    def get_token_str(self, token_id: int) -> str:
        """Get the string representation of a token ID.

        Args:
            token_id: The token ID.

        Returns:
            The token string, or empty string if not found.
        """
        return self.id_to_str.get(token_id, "")

    def get_token_id(self, token_str: str) -> Optional[int]:
        """Get the token ID for a string.

        Args:
            token_str: The token string.

        Returns:
            The token ID, or None if not found.
        """
        return self.str_to_id.get(token_str)


def is_valid_number_prefix(s: str) -> bool:
    """Check if string is a valid prefix of a JSON number.

    Handles negative numbers, decimals, and scientific notation.

    Args:
        s: The string to check.

    Returns:
        True if s could be the start of a valid JSON number.
    """
    if not s:
        return True
    i = 0
    n = len(s)
    if s[i] == '-':
        i += 1
        if i >= n:
            return True
    if i >= n:
        return True
    if s[i] == '0':
        i += 1
        if i < n and s[i].isdigit():
            return False
    elif s[i].isdigit():
        while i < n and s[i].isdigit():
            i += 1
    else:
        return False
    if i >= n:
        return True
    if s[i] == '.':
        i += 1
        if i >= n:
            return True
        if not s[i].isdigit():
            return False
        while i < n and s[i].isdigit():
            i += 1
    if i >= n:
        return True
    if s[i] in ('e', 'E'):
        i += 1
        if i >= n:
            return True
        if s[i] in ('+', '-'):
            i += 1
        if i >= n:
            return True
        if not s[i].isdigit():
            return False
        while i < n and s[i].isdigit():
            i += 1
    return i == n


def is_complete_number(s: str) -> bool:
    """Check if a string is a complete valid JSON number.

    Args:
        s: The string to check.

    Returns:
        True if s is a complete valid number.
    """
    if not s:
        return False
    try:
        float(s)
        return s[-1].isdigit()
    except ValueError:
        return False


class ConstrainedGenerator:
    """Generates structured JSON function calls using constrained decoding.

    The generator uses a template-based approach: it knows the exact
    JSON schema to produce and generates each segment (fixed structure
    vs free-form values) with appropriate constraints.

    The output schema is:
        {"name": "<function_name>", "parameters": {<key-value pairs>}}

    Args:
        llm: The Small_LLM_Model instance.
        vocabulary: Dict mapping token strings to token IDs.
        function_definitions: Available function definitions.
    """

    def __init__(
        self,
        llm: Any,
        vocabulary: Dict[str, int],
        function_definitions: List[FunctionDefinition],
    ) -> None:
        """Initialize the constrained generator.

        Args:
            llm: The LLM model instance.
            vocabulary: Token string to token ID mapping.
            function_definitions: List of available functions.
        """
        self.llm = llm
        self.vocab = TokenVocabulary(vocabulary)
        self.function_definitions = function_definitions
        self.func_by_name: Dict[str, FunctionDefinition] = {
            f.name: f for f in function_definitions
        }

    @staticmethod
    def _decode_token_str(token_str: str) -> str:
        """Decode a raw BPE token string to its actual text.

        BPE vocabularies use special characters like Ġ (U+0120)
        to represent spaces. This method converts them back.

        Args:
            token_str: The raw token string from vocabulary.

        Returns:
            The decoded text with BPE artifacts replaced.
        """
        # Ġ (U+0120) represents a space in GPT/Qwen BPE vocabularies
        result = token_str.replace('\u0120', ' ')
        # Ċ (U+010A) represents a newline
        result = result.replace('\u010a', '\n')
        # ĉ (U+0109) represents a tab
        result = result.replace('\u0109', '\t')
        return result

    def _build_prompt(
        self,
        user_prompt: str,
    ) -> str:
        """Build the LLM prompt including function descriptions.

        Args:
            user_prompt: The user's natural language query.

        Returns:
            Complete prompt string.
        """
        func_descs: List[str] = []
        for func in self.function_definitions:
            params = ", ".join(
                f"{k}: {v.type}"
                for k, v in func.parameters.items()
            )
            func_descs.append(
                f"- {func.name}({params}): {func.description}"
            )
        functions_text = "\n".join(func_descs)
        prompt = (
            "You are a function calling assistant. "
            "Given a user request, select the appropriate function "
            "and extract arguments.\n\n"
            f"Available functions:\n{functions_text}\n\n"
            f"User request: {user_prompt}\n\n"
            'Respond with JSON: {{"name": "<func>", '
            '"parameters": {{...}}}}\n'
            "Response: "
        )
        return prompt

    def _force_tokens(
        self,
        input_ids: List[int],
        text: str,
    ) -> List[int]:
        """Force-append tokens for a fixed string to input_ids.

        For deterministic parts of the JSON (structure characters),
        we can directly encode them without running the LLM.

        Args:
            input_ids: Current input IDs.
            text: The fixed text to append.

        Returns:
            Updated input_ids with the text tokens appended.
        """
        for char in text:
            token_id = self.vocab.get_token_id(char)
            if token_id is not None:
                input_ids.append(token_id)
            else:
                for t_str, t_id in self.vocab.str_to_id.items():
                    if t_str and char in t_str:
                        input_ids.append(t_id)
                        break
        return input_ids

    def _get_valid_continuation_ids(
        self,
        allowed_strings: List[str],
        generated_so_far: str,
    ) -> Set[int]:
        """Get token IDs that validly continue toward an allowed string.

        A token is valid if at least one allowed string has the
        token's text as a prefix of the remaining portion.

        Args:
            allowed_strings: Allowed complete string values.
            generated_so_far: What has been generated for this field.

        Returns:
            Set of valid token IDs.
        """
        valid: Set[int] = set()
        remaining_options = [
            s[len(generated_so_far):]
            for s in allowed_strings
            if s.startswith(generated_so_far)
        ]
        if not remaining_options:
            return valid
        for token_str, token_id in self.vocab.str_to_id.items():
            if not token_str:
                continue
            for remaining in remaining_options:
                if not remaining:
                    continue
                if remaining.startswith(token_str):
                    valid.add(token_id)
                    break
                if token_str.startswith(remaining):
                    valid.add(token_id)
                    break
        return valid

    def _get_valid_number_ids(
        self,
        num_so_far: str,
    ) -> Set[int]:
        """Get token IDs valid for continuing a number value.

        Args:
            num_so_far: Number text generated so far.

        Returns:
            Set of valid token IDs for number continuation.
        """
        valid: Set[int] = set()
        for token_str, token_id in self.vocab.str_to_id.items():
            if not token_str:
                continue
            candidate = num_so_far + token_str
            if is_valid_number_prefix(candidate):
                valid.add(token_id)
        return valid

    def _get_valid_string_content_ids(self) -> Set[int]:
        """Get token IDs valid as string content (no unescaped quotes).

        Checks the decoded form of the token to properly handle
        BPE artifacts like Ġ (space) and Ċ (newline).

        Returns:
            Set of valid token IDs for string content.
        """
        valid: Set[int] = set()
        for token_str, token_id in self.vocab.str_to_id.items():
            if not token_str:
                continue
            decoded = self._decode_token_str(token_str)
            is_ok = True
            in_escape = False
            for ch in decoded:
                if in_escape:
                    in_escape = False
                    continue
                if ch == '\\':
                    in_escape = True
                    continue
                if ch == '"':
                    is_ok = False
                    break
                if ord(ch) < 0x20:
                    is_ok = False
                    break
            if is_ok:
                valid.add(token_id)
        return valid

    def _apply_mask_and_select(
        self,
        input_ids: List[int],
        valid_ids: Set[int],
    ) -> int:
        """Get logits from LLM, mask invalid tokens, select best.

        Args:
            input_ids: Current input token IDs.
            valid_ids: Set of allowed token IDs.

        Returns:
            Selected token ID.

        Raises:
            RuntimeError: If no valid tokens available.
        """
        if not valid_ids:
            raise RuntimeError("No valid tokens available")

        logits = self.llm.get_logits_from_input_ids(input_ids)
        masked = np.full(len(logits), -math.inf)
        for tid in valid_ids:
            if tid < len(logits):
                masked[tid] = logits[tid]
        return int(np.argmax(masked))

    def _generate_from_options(
        self,
        input_ids: List[int],
        options: List[str],
        max_tokens: int = 64,
    ) -> Tuple[List[int], str]:
        """Generate text constrained to one of the given options.

        Used for selecting function names from a known list.

        Args:
            input_ids: Current input token IDs.
            options: Allowed string values.
            max_tokens: Maximum tokens to generate.

        Returns:
            Tuple of (updated input_ids, selected string).
        """
        generated = ""
        for _ in range(max_tokens):
            remaining = [
                o for o in options
                if o.startswith(generated)
            ]
            if len(remaining) == 1 and remaining[0] == generated:
                break
            if not remaining:
                break

            valid_ids = self._get_valid_continuation_ids(
                options, generated
            )
            if not valid_ids:
                break

            token_id = self._apply_mask_and_select(
                input_ids, valid_ids
            )
            token_str = self.vocab.get_token_str(token_id)
            generated += token_str
            input_ids.append(token_id)

            exact = [o for o in options if o == generated]
            if exact:
                still_possible = [
                    o for o in options
                    if o.startswith(generated) and o != generated
                ]
                if not still_possible:
                    break

        matched = [o for o in options if o == generated]
        if not matched:
            best = [o for o in options if o.startswith(generated)]
            if best:
                rest = best[0][len(generated):]
                for ch in rest:
                    tid = self.vocab.get_token_id(ch)
                    if tid is not None:
                        input_ids.append(tid)
                generated = best[0]
            else:
                generated = options[0]
                input_ids = list(input_ids)

        return input_ids, generated

    def _generate_number(
        self,
        input_ids: List[int],
        max_tokens: int = 32,
    ) -> Tuple[List[int], float]:
        """Generate a constrained number value.

        Args:
            input_ids: Current input token IDs.
            max_tokens: Maximum tokens.

        Returns:
            Tuple of (updated input_ids, parsed number value).
        """
        num_text = ""
        for _ in range(max_tokens):
            number_ids = self._get_valid_number_ids(num_text)
            end_ids: Set[int] = set()
            if is_complete_number(num_text):
                for t_str, t_id in self.vocab.str_to_id.items():
                    if t_str in (',', '}', ' '):
                        end_ids.add(t_id)
                valid_ids = number_ids | end_ids
            else:
                valid_ids = number_ids

            if not valid_ids:
                break

            token_id = self._apply_mask_and_select(
                input_ids, valid_ids
            )

            if token_id in end_ids:
                break

            token_str = self.vocab.get_token_str(token_id)
            num_text += token_str
            input_ids.append(token_id)

        if not num_text or not is_complete_number(num_text):
            return input_ids, 0.0

        return input_ids, float(num_text)

    def _generate_string(
        self,
        input_ids: List[int],
        max_tokens: int = 128,
    ) -> Tuple[List[int], str]:
        """Generate a constrained string value (without quotes).

        The quotes are handled by the caller; this generates the
        content between them.

        Args:
            input_ids: Current input token IDs.
            max_tokens: Maximum tokens.

        Returns:
            Tuple of (updated input_ids, string content).
        """
        content = ""
        string_content_ids = self._get_valid_string_content_ids()

        close_quote_ids: Set[int] = set()
        for t_str, t_id in self.vocab.str_to_id.items():
            if t_str == '"':
                close_quote_ids.add(t_id)

        for _ in range(max_tokens):
            if len(content) > 0:
                valid_ids = string_content_ids | close_quote_ids
            else:
                valid_ids = string_content_ids

            if not valid_ids:
                break

            token_id = self._apply_mask_and_select(
                input_ids, valid_ids
            )

            if token_id in close_quote_ids:
                break

            token_str = self.vocab.get_token_str(token_id)
            decoded_str = self._decode_token_str(token_str)
            content += decoded_str
            input_ids.append(token_id)

        return input_ids, content

    def _generate_boolean(
        self,
        input_ids: List[int],
    ) -> Tuple[List[int], bool]:
        """Generate a constrained boolean value.

        Args:
            input_ids: Current input token IDs.

        Returns:
            Tuple of (updated input_ids, boolean value).
        """
        input_ids, result = self._generate_from_options(
            input_ids, ["true", "false"]
        )
        return input_ids, result == "true"

    def _encode_fixed(self, text: str) -> List[int]:
        """Encode a fixed string into token IDs.

        Uses greedy matching from the vocabulary.

        Args:
            text: The string to encode.

        Returns:
            List of token IDs.
        """
        result = self.llm.encode(text)
        if hasattr(result, 'tolist'):
            ids: List[int] = result[0].tolist()
            return ids
        return list(result)

    def generate_function_call(
        self,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """Generate a complete function call from a user prompt.

        Produces a JSON object with 'name' and 'parameters' fields
        using constrained decoding to guarantee valid output.

        The generation process:
        1. Build prompt with function descriptions
        2. Generate '{"name": "' (fixed)
        3. Generate function name (constrained to known names)
        4. Generate '", "parameters": {' (fixed)
        5. For each parameter:
           a. Generate '"param_name": ' (fixed)
           b. Generate value (constrained by type)
           c. Generate separator (, or })
        6. Generate '}' (fixed)

        Args:
            user_prompt: The user's natural language request.

        Returns:
            Dict with 'name' and 'parameters' keys.

        Raises:
            RuntimeError: If generation fails.
        """
        full_prompt = self._build_prompt(user_prompt)
        input_ids: List[int] = self._encode_fixed(full_prompt)

        prefix = '{"name": "'
        prefix_ids = self._encode_fixed(prefix)
        input_ids.extend(prefix_ids)

        func_names = [f.name for f in self.function_definitions]
        input_ids, selected_name = self._generate_from_options(
            input_ids, func_names
        )

        if selected_name not in self.func_by_name:
            raise RuntimeError(
                f"Selected function '{selected_name}' "
                "not in definitions"
            )

        func_def = self.func_by_name[selected_name]
        param_names = list(func_def.parameters.keys())
        params_result: Dict[str, Any] = {}

        mid_section = '", "parameters": {'
        mid_ids = self._encode_fixed(mid_section)
        input_ids.extend(mid_ids)

        for i, pname in enumerate(param_names):
            if i > 0:
                sep = ", "
                sep_ids = self._encode_fixed(sep)
                input_ids.extend(sep_ids)

            ptype = func_def.parameters[pname].type

            key_str = f'"{pname}": '
            key_ids = self._encode_fixed(key_str)
            input_ids.extend(key_ids)

            if ptype == "number":
                input_ids, num_val = self._generate_number(input_ids)
                params_result[pname] = num_val

            elif ptype == "string":
                quote_ids = self._encode_fixed('"')
                input_ids.extend(quote_ids)

                input_ids, str_val = self._generate_string(input_ids)
                params_result[pname] = str_val

                close_ids = self._encode_fixed('"')
                input_ids.extend(close_ids)

            elif ptype == "boolean":
                input_ids, bool_val = self._generate_boolean(
                    input_ids
                )
                params_result[pname] = bool_val

            else:
                input_ids, other_val = self._generate_string(
                    input_ids
                )
                params_result[pname] = other_val

        close_ids = self._encode_fixed("}}")
        input_ids.extend(close_ids)

        return {
            "name": selected_name,
            "parameters": params_result,
        }
