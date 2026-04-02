from llm_sdk import Small_LLM_Model
from typing import Any, List
from .models import FunctionDefinition, PromptInput
import json
from functools import lru_cache


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
        self.funcs_ids: list[list[int]] = [self.llm.encode(
            func.name)[0].tolist() for func in self.functions.values()]
        self.functions_data = functions
        self.vocab_str = {
            k: v for k,
            v in self.llm._tokenizer.get_vocab().items()}
        self.vocab_id = {
            v: k for k,
            v in self.llm._tokenizer.get_vocab().items()}
        self.generate_output()

    @lru_cache(maxsize=None)
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

    @lru_cache(maxsize=None)
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

    @lru_cache(maxsize=None)
    def generate_output(self) -> None:
        """Generate structured output items for all prompts.

        Returns
        -------
        None
        """
        self._log_separator("LLM FUNCTION CALL GENERATION", self.BLUE)
        for prompt_index, user_request in enumerate(self.prompts, start=1):
            if user_request.prompt.strip() is None:
                self._log(
                    "SKIP",
                    f"Prompt #{prompt_index} has no text, skipping",
                    self.RED,
                )
                continue
            self._log(
                "PROMPT",
                f"#{prompt_index} {user_request.prompt}",
                self.CYAN,
            )
            prompt = self.generate_prompt(
                user_request.prompt, self.functions_data)
            name = '{"name": "'
            result = [prompt, name]

            self._log(
                "STEP 1",
                "Selecting function name with constrained decoding",
                self.BLUE,
            )
            func_name = self.generate_func_name(result)
            if func_name not in self.functions:
                raise ValueError(
                    f"Function {func_name} not found in definitions.")
            self._log("FUNCTION", f"Selected `{func_name}`", self.GREEN)

            func_data = [
                f for f in self.functions_data if f["name"] == func_name][0]
            result[0] = self.generate_prompt(user_request.prompt, func_data)
            result.append('", "parameters": ')

            self._log(
                "STEP 2",
                "Generating parameters with type constraints",
                self.BLUE,
            )
            self.generate_parameters(result, func_name)
            txt = "".join(result[1:])

            item = txt.strip()

            if not item.endswith("}}"):
                item += "}"

            try:
                parsed_item = json.loads(item)
                normalized_item = {
                    "prompt": user_request.prompt,
                    "name": func_name,
                    "parameters": parsed_item["parameters"]
                }
                self.output.append(normalized_item)
                self._log("OUTPUT", str(normalized_item), self.GREEN)
            except json.JSONDecodeError:
                self._log("ERROR", f"Skipped invalid JSON: {item}", self.RED)

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
            ids = self.llm.encode("".join(result) + value)[0].tolist()
            logits = self.llm.get_logits_from_input_ids(ids)
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
            token: Any = float(value.strip())
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
            logits[self.vocab_str.get(chars[0], -1)] = float("-inf")
            next_token = max(range(len(logits)), key=lambda i: logits[i])
        elif param_type == "boolean":
            allowed_tokens = set(self.llm.encode("true false"))
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
            ids = self.llm.encode("".join(result))[0].tolist()
            logits = self.llm.get_logits_from_input_ids(ids)
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
        return (
            f"Available functions:\n{functions_data}\n\n"
            f"User request: {prompt}\n\n"
            "Respond with a JSON object with keys 'name' and 'parameters'."
        )
