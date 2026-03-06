"""OpenAI API wrapper with retry logic and token tracking."""

import base64
import time
from pathlib import Path
from typing import Optional
from openai import OpenAI
from .config import ModelConfig


class LLMClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        if not config.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to .env file or set the environment variable.\n"
                "Folder management commands (create, list, info, health) don't need an API key."
            )
        self.client = OpenAI(api_key=config.api_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def complete(self, prompt, model=None, system_prompt=None, json_mode=False, temperature=None, max_tokens=4096):
        model = model or self.config.indexing_model
        temperature = temperature if temperature is not None else self.config.temperature
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self._call_with_retry(**kwargs)

    def complete_with_images(self, prompt, images, model=None, system_prompt=None, json_mode=False, temperature=None, max_tokens=4096, detail="high"):
        model = model or self.config.indexing_model
        temperature = temperature if temperature is not None else self.config.temperature
        content = []
        for img in images:
            b64 = self._encode_image(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": detail}})
        content.append({"type": "text", "text": prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self._call_with_retry(**kwargs)

    def _call_with_retry(self, **kwargs):
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                if response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                self.total_calls += 1
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"API failed after {self.config.max_retries} retries: {last_error}")

    def _encode_image(self, image):
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        if isinstance(image, Path) or (isinstance(image, str) and Path(image).exists()):
            return base64.b64encode(Path(image).read_bytes()).decode("utf-8")
        return str(image)

    @property
    def usage_summary(self):
        return {"total_calls": self.total_calls, "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens, "estimated_cost_usd": self._estimate_cost()}

    def _estimate_cost(self):
        return round((self.total_input_tokens / 1e6) * 2.50 + (self.total_output_tokens / 1e6) * 10.00, 4)
