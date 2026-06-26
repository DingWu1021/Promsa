# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Image Search Tool for VLM multi-turn training.

This is a DATA-DRIVEN tool - results come from the dataset, not from a live API.
The model calls image_search_tool() with no arguments, and the tool returns
pre-computed reverse image search results from the dataset.

Data format expected in tools_kwargs:
{
    "image_search_tool": {
        "create_kwargs": {
            "image_search_title_list": ["title1", "title2", ...],
            "image_search_thumbnail_list": ["path/to/thumb1.jpg", ...],
            "image_search_summary": "Optional summary text"
        }
    }
}
"""

import logging
import os
import re
import base64
import mimetypes
import asyncio
from typing import Any, Optional
from uuid import uuid4

import aiohttp

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ImageSearchTool(BaseTool):
    """
    A tool for reverse image search results.

    This is a DATA-DRIVEN tool - the search results are pre-computed and
    stored in the dataset. When the model calls this tool, it returns
    the pre-computed results including titles and thumbnail images.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize ImageSearchTool.

        Config options:
            use_image_search_summary: Include LLM summary in results (default: False)
            min_pixels: Min pixels for thumbnail resize
            max_pixels: Max pixels for thumbnail resize
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Config
        self.use_image_search_summary = config.get("use_image_search_summary", False)
        self.min_pixels = config.get("min_pixels", None)
        self.max_pixels = config.get("max_pixels", None)
        # Multi-call policy: allow up to 2 calls, each returns top-k unseen blocks.
        self.max_image_search_calls = int(config.get("max_image_search_calls", 2))
        self.image_search_return_k = int(config.get("image_search_return_k", 2))
        # Reuse the same summarizer settings/prompt family as existing text search stack.
        self.summarizer_base_url = os.getenv("SUMMARIZER_BASE_URL", "")
        self.summarizer_api_key = os.getenv("SUMMARIZER_API_KEY", "EMPTY")
        self.summarizer_model = config.get("summary_llm_model", "Qwen/Qwen3-VL-32B-Instruct")
        self.summarizer_timeout = config.get("summary_timeout", 120)
        self.summarizer_content_limit = config.get("summary_content_limit", 100000)
        self.summarizer_max_tokens = config.get("summary_max_tokens", 1024)
        self.summary_max_attempts = int(config.get("summary_max_attempts", 3))
        self.summary_retry_delay = float(config.get("summary_retry_delay", 1.0))
        self.summary_failure_text = config.get(
            "summary_failure_text", "This tool call failed for this round."
        )

        logger.info(f"Initialized ImageSearchTool: use_summary={self.use_image_search_summary}")

    def _build_image_data_url(self, image_path: str) -> Optional[str]:
        """Build a data URL for one local image path; return None on failure."""
        if not isinstance(image_path, str) or not image_path.strip():
            return None
        image_path = image_path.strip()
        if not os.path.isfile(image_path):
            return None
        try:
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "image/jpeg"
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            logger.warning(f"Failed to read image for image summary context: {e}")
            return None

    async def _summarize_block(self, query_hint: str, block_text: str, image_path: Optional[str] = None) -> Optional[str]:
        """Summarize one evidence block with retries; return None if all attempts fail."""
        if not isinstance(block_text, str):
            return None
        block_text = block_text.strip()
        if not block_text:
            return None
        if not self.summarizer_base_url:
            return None

        system_prompt = (
            "You are a helpful assistant. Your task is to summarize the main content of the given web page "
            "in no more than four sentences. Your summary should cover the overall key points of the page, "
            "not just parts related to the user's question.\n\n"
            "If any part of the content is helpful for answering the user's question, prioritize it and state "
            "it explicitly in the summary. Focus on answer-relevant evidence, and keep the summary concise, "
            "factual, and informative."
        )
        user_prompt = (
            f"Webpage Content (first {self.summarizer_content_limit} characters) is: "
            f"{block_text[: self.summarizer_content_limit]}\n"
            f"Question: {query_hint or 'N/A'}"
        )

        user_content: Any = user_prompt
        image_data_url = self._build_image_data_url(image_path) if image_path else None
        if image_data_url:
            # Keep the same summary prompt, but provide image context to VL model.
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]

        payload = {
            "model": self.summarizer_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self.summarizer_max_tokens,
        }
        if "qwen3" in self.summarizer_model.lower():
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        timeout = aiohttp.ClientTimeout(total=self.summarizer_timeout)
        url = self.summarizer_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.summarizer_api_key}",
            "Content-Type": "application/json",
        }

        attempts = max(1, self.summary_max_attempts)
        for attempt in range(1, attempts + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        response.raise_for_status()
                        data = await response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("summarizer content is not a string")
                # Keep consistent with existing web_search_server cleaner.
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
                if not content:
                    raise ValueError("summarizer content is empty")
                return content
            except Exception as e:
                logger.warning(
                    "Image block summarization failed (attempt %s/%s): %s",
                    attempt,
                    attempts,
                    e,
                )
                if attempt < attempts:
                    await asyncio.sleep(max(0.0, self.summary_retry_delay))
        return None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Create a tool instance with pre-computed search results.

        Args:
            instance_id: Optional unique identifier
            **kwargs: Should contain create_kwargs with:
                - image_search_title_list: List of result titles
                - image_search_summary: Optional merged summary text
                - image_search_text_blocks: Optional per-document text blocks

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Extract search results from kwargs
        title_list = kwargs.get("image_search_title_list", []) or []
        summary = kwargs.get("image_search_summary", None)
        text_blocks = kwargs.get("image_search_text_blocks", None)
        question_hint = kwargs.get("question_hint", "")
        image_path = kwargs.get("image_path", None)

        normalized_text_blocks = []
        if isinstance(text_blocks, list):
            for block in text_blocks:
                if isinstance(block, str):
                    block = block.strip()
                    if block:
                        normalized_text_blocks.append(block)
        elif isinstance(text_blocks, str):
            block = text_blocks.strip()
            if block:
                normalized_text_blocks.append(block)

        self._instance_dict[instance_id] = {
            "title_list": title_list,
            "summary": summary,
            "all_text_blocks": normalized_text_blocks,
            "question_hint": question_hint,
            "image_path": image_path,
            "call_count": 0,
            "served_block_indices": set(),
            "called": False,
        }

        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute image search and return text-only evidence in chunked format.

        Args:
            instance_id: Instance ID from create()
            parameters: Tool parameters (typically empty for image_search)
            **kwargs: Additional kwargs

        Returns:
            Tuple of (ToolResponse with chunked text evidence, reward, info_dict)
        """
        instance_data = self._instance_dict.get(instance_id, {})
        title_list = instance_data.get("title_list", []) or []
        summary = instance_data.get("summary", None)
        all_text_blocks = instance_data.get("all_text_blocks", [])
        default_question_hint = instance_data.get("question_hint", "") or ""
        image_path = instance_data.get("image_path", None)

        # Mark as called
        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["called"] = True
            self._instance_dict[instance_id]["call_count"] = (
                self._instance_dict[instance_id].get("call_count", 0) + 1
            )

        call_count = instance_data.get("call_count", 0)
        if call_count > self.max_image_search_calls:
            return (
                ToolResponse(text="Image search call limit reached."),
                0.0,
                {"success": True, "num_results": 0, "call_count": call_count, "limit_reached": True},
            )

        # Text-only mode with unseen-block filtering:
        # each call returns image_search_return_k blocks not returned in prior calls.
        # Note: <tool_response> tags are added by the chat template, not here.
        served_indices = instance_data.get("served_block_indices", set())
        candidate_pairs = []
        if isinstance(all_text_blocks, list):
            for i, block in enumerate(all_text_blocks):
                if i in served_indices:
                    continue
                candidate_pairs.append((i, block))

        selected_pairs = candidate_pairs[: self.image_search_return_k]
        normalized_blocks = [b for _, b in selected_pairs]
        if normalized_blocks:
            query_hint = ""
            if isinstance(parameters, dict):
                query_hint = parameters.get("query", "") or parameters.get("question", "") or ""
            if not query_hint:
                query_hint = default_question_hint
            summarized_blocks = []
            for block in normalized_blocks:
                summarized = await self._summarize_block(query_hint, block, image_path=image_path)
                if summarized is None:
                    return (
                        ToolResponse(text=self.summary_failure_text),
                        0.0,
                        {
                            "success": False,
                            "num_results": 0,
                            "call_count": call_count,
                            "remaining_blocks": len(candidate_pairs),
                            "error": "image_summary_failed",
                            "limit_reached": False,
                        },
                    )
                summarized_blocks.append(summarized)

            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["served_block_indices"] = (
                    set(served_indices) | {i for i, _ in selected_pairs}
                )

            interleaved_content = [{"type": "text", "text": "Reverse Image Search Results:"}]
            for block in summarized_blocks:
                # Keep output purely as summarized evidence blocks (no title prefix).
                interleaved_content.append({"type": "text", "text": f"\n\n{block}"})
            return (
                ToolResponse(interleaved_content=interleaved_content),
                0.0,
                {
                    "success": True,
                    "num_results": len(normalized_blocks),
                    "call_count": call_count,
                    "remaining_blocks": len(candidate_pairs) - len(selected_pairs),
                    "limit_reached": False,
                },
            )

        summary_text = summary.strip() if isinstance(summary, str) else ""
        if not summary_text:
            return (
                ToolResponse(text="No relevant information found."),
                0.0,
                {"success": True, "num_results": 0, "limit_reached": False},
            )

        return (
            ToolResponse(interleaved_content=[{"type": "text", "text": summary_text}]),
            0.0,
            {"success": True, "num_results": len(title_list), "limit_reached": False},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
