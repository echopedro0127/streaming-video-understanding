from __future__ import annotations

import time

from .base import ModelRequest, ModelResponse
from .prompt import build_prompt, parse_answer_letter


class SmolVLMBackend:
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        max_new_tokens: int = 8,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "SmolVLMBackend requires torch and transformers; install requirements.txt first"
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens

    def infer(self, request: ModelRequest) -> ModelResponse:
        torch = self._torch
        prompt = build_prompt(
            question=request.question,
            options=request.options,
            timestamps_s=request.timestamps_s,
            query_time_s=request.query_time_s,
            include_timestamps=request.include_timestamps,
        )
        content = [{"type": "image", "image": image} for image in request.images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        started = time.perf_counter()
        processor_latency_ms: float | None = None
        generation_latency_ms: float | None = None
        postprocess_latency_ms: float | None = None
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            processor_started = time.perf_counter()
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            processor_latency_ms = (time.perf_counter() - processor_started) * 1000.0

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_started = time.perf_counter()
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_latency_ms = (time.perf_counter() - generation_started) * 1000.0

            postprocess_started = time.perf_counter()
            input_length = inputs["input_ids"].shape[-1]
            generated_text = self.processor.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
            )[0]
            answer = parse_answer_letter(generated_text)
            postprocess_latency_ms = (time.perf_counter() - postprocess_started) * 1000.0
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"SmolVLM inference ran out of memory for {len(request.images)} frames"
            ) from exc

        return ModelResponse(
            raw_text=generated_text,
            answer=answer,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            processor_latency_ms=processor_latency_ms,
            generation_latency_ms=generation_latency_ms,
            postprocess_latency_ms=postprocess_latency_ms,
        )
