from llm_sdk import Small_LLM_Model
from typing import Any, List
from .models import FunctionDefinition, PromptInput
import json


class JsonStructure():
    """Generate structured function-calling outputs with constraints.

    Parameters
    ----------
    output : list[dict[str, Any]]
        Mutable list where generated output rows are appended.
    functions_definition : list[FunctionDefinition]
        Validated function schema definitions.
    functions : list[dict[str, Any]]
        Raw function-definition payload used in prompt context.
    prompts : list[PromptInput]
        Validated user prompts to process.
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"

    def __init__(
            self,
            output: List[dict[str, Any]],
            functions_definition: List[FunctionDefinition],
            functions: List[dict[str, Any]],
            prompts: List[PromptInput]) -> None:
        """Initialize generator state and start output generation.

        Parameters
        ----------
        output : list[dict[str, Any]]
            Mutable output list to populate.
        functions_definition : list[FunctionDefinition]
            Parsed function definitions.
        functions : list[dict[str, Any]]
            Raw function metadata for prompt construction.
        prompts : list[PromptInput]
            Parsed user prompts.

        Returns
        -------
        None
        """
        self.output = output
        self.llm = Small_LLM_Model()
        self.prompts = prompts
        self.functions = {func.name: func for func in functions_definition}
        self.funcs_ids: list[list[int]] = [
            self.llm.encode_ids(func.name) for func in self.functions.values()
        ]
        self.functions_data = functions
        self.function_data_by_name = {
            function_data["name"]: function_data
            for function_data in self.functions_data
            if "name" in function_data
        }
        self.token_to_id = self.llm.get_vocab()
        self.batch_size = 4
        self.max_function_attempts = 3
        self.max_value_steps = 80
        self._encode_cache: dict[str, list[int]] = {}
        self._logits_cache: dict[tuple[int, ...], list[float]] = {}
        self._prompt_cache: dict[tuple[str, str], str] = {}
        self._bool_allowed_tokens = set(self.llm.encode_ids("true false"))
        self.generate_output()

    def _encode_cached(self, text: str) -> list[int]:
        """Encode text with memoization.

        Parameters
        ----------
        text : str
            Input text to encode.

        Returns
        -------
        list[int]
            Encoded token ids.
        """
        cached = self._encode_cache.get(text)
        if cached is not None:
            return cached
        encoded = self.llm.encode_ids(text)
        self._encode_cache[text] = encoded
        return encoded

    def _get_logits_cached(self, input_ids: list[int]) -> list[float]:
        """Get logits for input ids with memoization.

        Parameters
        ----------
        input_ids : list[int]
            Token ids used as model input.

        Returns
        -------
        list[float]
            Next-token logits.
        """
        key = tuple(input_ids)
        cached = self._logits_cache.get(key)
        if cached is not None:
            return list(cached)
        logits = self.llm.get_logits_from_input_ids(input_ids)
        self._logits_cache[key] = logits
        return list(logits)

    def _iter_prompt_batches(
        self,
        prompts: list[PromptInput],
    ) -> list[list[PromptInput]]:
        """Split prompts into fixed-size batches.

        Parameters
        ----------
        prompts : list[PromptInput]
            Prompt list to batch.

        Returns
        -------
        list[list[PromptInput]]
            Prompt batches in input order.
        """
        batches: list[list[PromptInput]] = []
        for start in range(0, len(prompts), self.batch_size):
            batches.append(prompts[start:start + self.batch_size])
        return batches

    def _log(self, stage: str, message: str, color: str) -> None:
        """Print one colorized log line for generation tracing.

        Parameters
        ----------
        stage : str
            Short stage name shown in brackets.
        message : str
            Human-readable log message.
        color : str
            ANSI color escape sequence.

        Returns
        -------
        None
        """
        label = f"{self.BOLD}{color}[{stage}]{self.RESET}"
        print(f"{label} {message}")

    def _log_separator(self, title: str, color: str) -> None:
        """Print a colorized section separator.

        Parameters
        ----------
        title : str
            Title displayed in the separator.
        color : str
            ANSI color escape sequence.

        Returns
        -------
        None
        """
        line = "=" * 60
        print(
            f"{self.BOLD}{color}{line}\n{title}\n{line}{self.RESET}"
        )

    def _default_value_for_type(self, param_type: str) -> Any:
        """Return a safe default value for a parameter type.

        Parameters
        ----------
        param_type : str
            Parameter type string.

        Returns
        -------
        Any
            Default value compatible with the expected type.
        """
        if param_type == "number":
            return 0.0
        if param_type == "integer":
            return 0
        if param_type == "boolean":
            return False
        return ""

    def _build_fallback_parameters(self, func_name: str) -> dict[str, Any]:
        """Build fallback parameters for a function.

        Parameters
        ----------
        func_name : str
            Function name to build defaults for.

        Returns
        -------
        dict[str, Any]
            Parameter object filled with safe defaults.
        """
        function_definition = self.functions[func_name]
        fallback: dict[str, Any] = {}
        for (
            parameter_name,
            parameter_def,
        ) in function_definition.parameters.items():
            fallback[parameter_name] = self._default_value_for_type(
                parameter_def.type
            )
        return fallback

    def _recover_function_name(self, candidate: str) -> str:
        """Recover a valid function name from an invalid candidate.

        Parameters
        ----------
        candidate : str
            Generated function name candidate.

        Returns
        -------
        str
            A valid function name from known definitions.
        """
        if candidate in self.functions:
            return candidate

        for function_name in self.functions:
            if function_name.startswith(candidate):
                return function_name
            if candidate.startswith(function_name):
                return function_name

        first_function = next(iter(self.functions.keys()))
        self._log(
            "RECOVERY",
            f"Fallback to `{first_function}` for invalid `{candidate}`",
            self.YELLOW,
        )
        return first_function

    def _normalize_output_item(
        self,
        prompt: str,
        func_name: str,
        raw_payload: str,
    ) -> dict[str, Any]:
        """Parse and normalize a generated output payload.

        Parameters
        ----------
        prompt : str
            Original user prompt.
        func_name : str
            Selected function name.
        raw_payload : str
            Raw generated JSON text.

        Returns
        -------
        dict[str, Any]
            Normalized output object with recovery fallback when needed.
        """
        try:
            parsed_item = json.loads(raw_payload)
            parameters = parsed_item.get("parameters", {})
            if not isinstance(parameters, dict):
                raise ValueError("Parameters payload is not an object")
            return {
                "prompt": prompt,
                "name": func_name,
                "parameters": parameters,
            }
        except (json.JSONDecodeError, ValueError) as error:
            self._log(
                "RECOVERY",
                f"Using fallback parameters after parse failure: {error}",
                self.YELLOW,
            )
            return {
                "prompt": prompt,
                "name": func_name,
                "parameters": self._build_fallback_parameters(func_name),
            }

    def generate_output(self) -> None:
        """Generate structured output items for all prompts.

        Returns
        -------
        None
        """
        self._log_separator("LLM FUNCTION CALL GENERATION", self.BLUE)
        prompt_index = 0
        for batch_id, prompt_batch in enumerate(
            self._iter_prompt_batches(self.prompts),
            start=1,
        ):
            self._log(
                "BATCH",
                (
                    f"Processing batch #{batch_id} with "
                    f"{len(prompt_batch)} prompt(s)"
                ),
                self.BLUE,
            )
            for user_request in prompt_batch:
                prompt_index += 1
                if not user_request.prompt.strip():
                    self._log(
                        "SKIP",
                        f"Prompt #{prompt_index} has no text, skipping",
                        self.RED,
                    )
                    continue
                try:
                    self._log(
                        "PROMPT",
                        f"#{prompt_index} {user_request.prompt}",
                        self.CYAN,
                    )

                    self._log(
                        "STEP 1",
                        "Selecting function name with constrained decoding",
                        self.BLUE,
                    )

                    generated_name = ""
                    base_prompt = self.generate_prompt(
                        user_request.prompt,
                        self.functions_data,
                    )
                    for attempt in range(1, self.max_function_attempts + 1):
                        selection_result = [base_prompt, '{"name": "']
                        generated_name = self.generate_func_name(
                            selection_result
                        )
                        if generated_name in self.functions:
                            break
                        self._log(
                            "RETRY",
                            (
                                f"Selection attempt {attempt} failed: "
                                f"`{generated_name}`"
                            ),
                            self.YELLOW,
                        )

                    func_name = self._recover_function_name(
                        generated_name
                    )
                    self._log(
                        "FUNCTION",
                        f"Selected `{func_name}`",
                        self.GREEN,
                    )

                    func_data = self.function_data_by_name.get(
                        func_name,
                        {},
                    )
                    result = [
                        self.generate_prompt(user_request.prompt, func_data),
                        f'{{"name": "{func_name}',
                    ]
                    result.append('", "parameters": ')

                    self._log(
                        "STEP 2",
                        "Generating parameters with type constraints",
                        self.BLUE,
                    )
                    self.generate_parameters(result, func_name)
                    item = "".join(result[1:]).strip()
                    if not item.endswith("}}"):
                        item += "}"

                    normalized_item = self._normalize_output_item(
                        prompt=user_request.prompt,
                        func_name=func_name,
                        raw_payload=item,
                    )
                    self.output.append(normalized_item)
                    self._log("OUTPUT", str(normalized_item), self.GREEN)
                except Exception as error:
                    self._log(
                        "ERROR",
                        f"Prompt #{prompt_index} failed: {error}",
                        self.RED,
                    )
                    fallback_function = next(iter(self.functions.keys()))
                    fallback_item = {
                        "prompt": user_request.prompt,
                        "name": fallback_function,
                        "parameters": self._build_fallback_parameters(
                            fallback_function
                        ),
                    }
                    self.output.append(fallback_item)
                    self._log(
                        "RECOVERY",
                        f"Fallback output appended: {fallback_item}",
                        self.YELLOW,
                    )

    def generate_parameters(self, result: list[str], func_name: str) -> None:
        """Generate a JSON parameter object for one function.

        Parameters
        ----------
        result : list[str]
            Mutable token/text buffer representing generated content.
        func_name : str
            Selected function name whose parameters are generated.

        Returns
        -------
        None
        """
        result.append('{')
        parameters = self.functions[func_name].parameters.items()
        self._log(
            "PARAMETERS",
            (
                "Preparing "
                f"{len(self.functions[func_name].parameters)} parameter(s)"
            ),
            self.MAGENTA,
        )
        for i, (param_name, param_def) in enumerate(parameters):
            self._log(
                "PARAM",
                f"{param_name} (type={param_def.type})",
                self.MAGENTA,
            )
            result.append(f'"{param_name}": ')
            if self.get_value(
                    result,
                    param_def.type,
                    i + 1 == len(parameters)):
                break

    def get_value(self, result: list[str], param_type: str, end: bool) -> bool:
        """Generate one constrained parameter value.

        Parameters
        ----------
        result : list[str]
            Mutable token/text buffer representing generated content.
        param_type : str
            Expected parameter type (e.g. ``string``, ``number``).
        end : bool
            Whether this is the final parameter in the JSON object.

        Returns
        -------
        bool
            ``True`` when JSON closure is complete, otherwise ``False``.
        """
        value = ""
        step = 0
        while True:
            step += 1
            if step > self.max_value_steps:
                self._log(
                    "RECOVERY",
                    "Max token steps reached, using default value",
                    self.YELLOW,
                )
                tail = "}" if end else ","
                self.handle_value(result, "", param_type, tail)
                return end
            ids = self._encode_cached("".join(result) + value)
            logits = self._get_logits_cached(ids)
            next_token = self.constrained_decoding(logits, param_type, end)
            txt = self.llm.decode([next_token])

            print(f"{self.YELLOW}{'-' * 50}{self.RESET}")
            data = "".join(result[1:] + [value] + [txt])
            self._log("TOKEN", f"Step {step} token={txt!r}", self.YELLOW)
            self._log("TYPE", f"Expected type={param_type}", self.YELLOW)
            self._log("PARTIAL", data, self.CYAN)
            print(f"{self.YELLOW}{'-' * 50}{self.RESET}")

            tmp = value.rstrip()
            if param_type == "string" and tmp.endswith('",'):
                self._log(
                    "DECISION",
                    "Detected completed string segment",
                    self.GREEN,
                )
                self.handle_value(result, value, param_type, "")
                return False
            elif param_type != "string" and any(i == "," for i in txt):
                self._log(
                    "DECISION",
                    "Detected parameter separator token",
                    self.GREEN,
                )
                self.handle_value(result, value, param_type, txt)
                return False
            try:
                tmp = txt.rstrip()
                if tmp.endswith("}}"):
                    txt = txt.rstrip()
                    self._log(
                        "DECISION",
                        "Detected end of JSON object",
                        self.GREEN,
                    )
                    txt = txt[:-1]

                verify_json = "".join(result[result.index(
                    '", "parameters": ') + 1:] + [value] + [txt])
                self._log("VERIFY", verify_json, self.BLUE)
                parsed = json.loads(verify_json)
                if isinstance(parsed, dict):
                    self.handle_value(result, value, param_type, txt)
                    self._log(
                        "DECISION",
                        "JSON validation succeeded",
                        self.GREEN,
                    )
                    return True
            except json.JSONDecodeError:
                self._log(
                    "RETRY",
                    "JSON still incomplete, continuing generation",
                    self.RED,
                )
                value = value + txt
                continue
            return False

    def handle_value(
        self,
        result: list[str],
        value: str,
        param_type: str,
        txt: str,
    ) -> None:
        """Normalize and append a generated parameter value.

        Parameters
        ----------
        result : list[str]
            Mutable token/text buffer representing generated content.
        value : str
            Accumulated raw value text.
        param_type : str
            Target parameter type.
        txt : str
            Trailing token text to append after normalization.

        Returns
        -------
        None
        """
        if param_type == "number":
            try:
                token: Any = float(value.strip())
            except ValueError:
                token = self._default_value_for_type(param_type)
                self._log(
                    "RECOVERY",
                    "Invalid number token, using default 0.0",
                    self.YELLOW,
                )
        elif param_type == "integer":
            try:
                token = int(value.strip())
            except ValueError:
                token = self._default_value_for_type(param_type)
                self._log(
                    "RECOVERY",
                    "Invalid integer token, using default 0",
                    self.YELLOW,
                )
        elif param_type == "boolean":
            token = value.strip().lower() == "true"
        else:
            token = value.strip()

        result.append(str(token) + txt)

    def constrained_decoding(
        self,
        logits: list[float],
        param_type: str,
        end: bool,
    ) -> int:
        """Select next token while enforcing type-level constraints.

        Parameters
        ----------
        logits : list[float]
            Raw next-token logits.
        param_type : str
            Target parameter type.
        end : bool
            Whether the current value is for the final parameter.

        Returns
        -------
        int
            Selected next token id.

        Raises
        ------
        ValueError
            If ``param_type`` is unsupported.
        """
        next_token = 0
        if param_type == "number" or param_type == "integer":
            chars = [".", "-", ","]
            if end:
                chars = [".", "-", "}"]
            if param_type == "integer":
                chars.remove(".")

            while True:
                next_token = max(range(len(logits)), key=lambda i: logits[i])
                txt = self.llm.decode([next_token])
                if not txt.isdigit() and not any(c in txt for c in chars):
                    logits[next_token] = float("-inf")
                else:
                    break
        elif param_type == "string":
            chars = ["}"]
            if end:
                chars = [","]
            token_id = self.token_to_id.get(chars[0], -1)
            if token_id >= 0:
                logits[token_id] = float("-inf")
            next_token = max(range(len(logits)), key=lambda i: logits[i])
        elif param_type == "boolean":
            allowed_tokens = set(self.llm.encode_ids("true false"))
            while True:
                next_token = max(range(len(logits)), key=lambda i: logits[i])
                if next_token not in allowed_tokens:
                    logits[next_token] = float("-inf")
                else:
                    break
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
        return next_token

    def generate_func_name(self, result: list[str]) -> str:
        """Generate a valid function name from allowed definitions.

        Parameters
        ----------
        result : list[str]
            Mutable token/text buffer representing generated content.

        Returns
        -------
        str
            Generated function name.
        """
        func_logits: list[int] = []
        self._log(
            "FUNC",
            "Starting constrained function-name decoding",
            self.BLUE,
        )
        while True:
            ids = self._encode_cached("".join(result))
            logits = self._get_logits_cached(ids)
            next_token = self.constrained_decoding_for_func_names(
                func_logits, logits)
            if not next_token:
                break

            token_text = self.llm.decode([next_token])
            self._log(
                "FUNC TOKEN",
                f"token={token_text!r} id={next_token}",
                self.MAGENTA,
            )
            result.append(token_text)
            func_logits.append(next_token)
        return self.llm.decode(func_logits)

    def constrained_decoding_for_func_names(
        self,
        func_logits: list[int],
        logits: list[float],
    ) -> int | None:
        """Restrict function-name generation to valid continuations.

        Parameters
        ----------
        func_logits : list[int]
            Already generated token ids for the function name.
        logits : list[float]
            Raw next-token logits.

        Returns
        -------
        int | None
            Next valid token id, or ``None`` if no continuation exists.
        """
        available_tokens: set[int] = set()
        n = len(func_logits)
        for func_ids in self.funcs_ids:
            if len(func_ids) > n and func_ids[:n] == func_logits:
                for id_ in func_ids[n:]:
                    available_tokens.add(id_)

        if len(available_tokens) == 0:
            return None
        while True:
            next_token = max(range(len(logits)), key=lambda i: logits[i])
            if next_token not in available_tokens:
                logits[next_token] = float("-inf")
            else:
                break
        return next_token

    def generate_prompt(self, prompt: str, functions_data: Any) -> str:
        """Build the LLM prompt used for constrained generation.

        Parameters
        ----------
        prompt : Any
            User prompt or prompt text.
        functions_data : Any
            Function metadata included in context.

        Returns
        -------
        str
            Rendered prompt text.
        """
        cache_key = (prompt, json.dumps(functions_data, sort_keys=True))
        cached_prompt = self._prompt_cache.get(cache_key)
        if cached_prompt is not None:
            return cached_prompt

        rendered_prompt = (
            f"Available functions:\n{functions_data}\n\n"
            f"User request: {prompt}\n\n"
            "Respond with a JSON object with keys 'name' and 'parameters'."
        )
        self._prompt_cache[cache_key] = rendered_prompt
        return rendered_prompt
