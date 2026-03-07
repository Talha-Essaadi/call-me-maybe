import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .constrained_decoder import ConstrainedGenerator
from .models import FunctionCallResult, FunctionDefinition, PromptInput


class FunctionCallingPipeline:
    """Pipeline for processing prompts into function calls.

    Manages the LLM model lifecycle and processes batches of
    prompts through the constrained decoder.

    Args:
        function_definitions: List of available function definitions.
        model_name: HuggingFace model identifier.
    """

    def __init__(
        self,
        function_definitions: List[FunctionDefinition],
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Initialize the function calling pipeline.

        Args:
            function_definitions: Available function definitions.
            model_name: Name of the model to load.
        """
        self.function_definitions = function_definitions
        self.model_name = model_name
        self.llm: Any = None
        self.vocabulary: Dict[str, int] = {}
        self.generator: Any = None

    def initialize(self) -> None:
        """Initialize the LLM model and vocabulary.

        Loads the model, retrieves the vocabulary file, and
        creates the constrained generator.

        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            from llm_sdk import Small_LLM_Model
            print(f"Loading model: {self.model_name}...")
            start = time.time()
            self.llm = Small_LLM_Model(model_name=self.model_name)
            elapsed = time.time() - start
            print(f"Model loaded in {elapsed:.1f}s")

            print("Loading vocabulary...")
            vocab_path = self.llm.get_path_to_vocab_file()
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocabulary = json.load(f)
            print(
                f"Vocabulary loaded: {len(self.vocabulary)} tokens"
            )

            self.generator = ConstrainedGenerator(
                llm=self.llm,
                vocabulary=self.vocabulary,
                function_definitions=self.function_definitions,
            )
            print("Pipeline initialized successfully")

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize pipeline: {e}"
            ) from e

    def process_prompt(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        """Process a single prompt into a function call.

        Args:
            prompt: The natural language prompt.

        Returns:
            Dict with 'prompt', 'name', and 'parameters' keys.

        Raises:
            RuntimeError: If processing fails.
        """
        if self.generator is None:
            raise RuntimeError(
                "Pipeline not initialized. Call initialize() first."
            )

        try:
            result = self.generator.generate_function_call(prompt)
            return {
                "prompt": prompt,
                "name": result["name"],
                "parameters": result["parameters"],
            }
        except Exception as e:
            raise RuntimeError(
                f"Failed to process prompt '{prompt}': {e}"
            ) from e

    def process_all(
        self,
        prompts: List[PromptInput],
    ) -> List[Dict[str, Any]]:
        """Process all prompts and return results.

        Args:
            prompts: List of PromptInput objects to process.

        Returns:
            List of result dictionaries.
        """
        results: List[Dict[str, Any]] = []
        total = len(prompts)

        print(f"\nProcessing {total} prompts...")
        start_time = time.time()

        for i, prompt_obj in enumerate(prompts, 1):
            prompt_text = prompt_obj.prompt
            print(f"\n[{i}/{total}] Processing: {prompt_text}")

            try:
                result = self.process_prompt(prompt_text)
                results.append(result)
                print(
                    f"  -> {result['name']}"
                    f"({result['parameters']})"
                )
            except RuntimeError as e:
                print(f"  ERROR: {e}")
                results.append({
                    "prompt": prompt_text,
                    "name": "error",
                    "parameters": {},
                })

        elapsed = time.time() - start_time
        print(f"\nProcessed {total} prompts in {elapsed:.1f}s")
        print(
            f"Success rate: "
            f"{sum(1 for r in results if r['name'] != 'error')}"
            f"/{total}"
        )

        return results

    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Save results to a JSON file.

        Args:
            results: List of result dictionaries.
            output_path: Path to write the output JSON file.

        Raises:
            IOError: If writing fails.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            validated: List[Dict[str, Any]] = []
            for r in results:
                try:
                    result_obj = FunctionCallResult(**r)
                    validated.append(result_obj.model_dump())
                except Exception:
                    validated.append(r)

            with open(
                output_path, "w", encoding="utf-8"
            ) as f:
                json.dump(validated, f, indent=2, ensure_ascii=False)

            print(f"\nResults saved to: {output_path}")

        except Exception as e:
            raise IOError(
                f"Failed to save results to {output_path}: {e}"
            ) from e
