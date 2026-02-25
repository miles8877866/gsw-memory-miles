"""
GSW Generation Operator.

This module contains the GSWOperator class for generating Generative Semantic
Workspaces from text using sophisticated semantic role extraction.
"""

import json
from typing import Any, Dict, Optional, Union
from bespokelabs import curator


from ...prompts.operator_prompts import FactualExtractionPrompts, OperatorPrompts, PromptType
from ..models import GSWStructure


class GSWOperator(curator.LLM):
    """Curator class for generating GSWs using sophisticated semantic role extraction."""

    require_all_responses = False
    # we don't set response_format here to maintain maximum manual parsing flexibility
    # but the constructor in operator.py might still pass it.

    def __init__(self, prompt_type: PromptType = PromptType.EPISODIC, **kwargs):
        """Initialize GSWOperator with specified prompt type.

        Args:
            prompt_type: Type of prompts to use (EPISODIC or FACTUAL)
            **kwargs: Additional arguments passed to curator.LLM
        """
        # Pop parameters that curator doesn't like in __init__
        kwargs.pop("max_concurrent_requests", None)
        kwargs.pop("require_all_responses", None)
        
        super().__init__(**kwargs)
        self.require_all_responses = False
        self.prompt_type = prompt_type

        # Select appropriate prompt class based on type
        if prompt_type == PromptType.EPISODIC:
            self.prompt_class = OperatorPrompts
        elif prompt_type == PromptType.FACTUAL:
            self.prompt_class = FactualExtractionPrompts
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

    def prompt(self, input):
        """Create a prompt for the LLM to generate a GSW."""
        return [
            {"role": "system", "content": self.prompt_class.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.prompt_class.USER_PROMPT_TEMPLATE.format(
                    input_text=input["text"],
                    background_context=input.get("context", "")
                ),
            },
        ]

    def parse(self, input, response):
        """Robustly parse the LLM response into a GSW structure."""
        
        # Handle case where response is already a GSWStructure (due to response_format)
        if isinstance(response, GSWStructure):
            gsw_dict = response.model_dump()
        else:
            # It's likely a raw response (dict or string)
            content = ""
            if isinstance(response, str):
                content = response
            elif isinstance(response, dict) and "choices" in response:
                content = response["choices"][0]["message"]["content"]
            elif hasattr(response, "choices"):
                content = response.choices[0].message.content
            else:
                content = str(response)

            # Try to extract JSON
            gsw_dict = {"entity_nodes": [], "verb_phrase_nodes": []}
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    gsw_dict = data
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Final safety check for required fields
        if not isinstance(gsw_dict, dict):
            gsw_dict = {"entity_nodes": [], "verb_phrase_nodes": []}
            
        if "entity_nodes" not in gsw_dict:
            gsw_dict["entity_nodes"] = []
        if "verb_phrase_nodes" not in gsw_dict:
            gsw_dict["verb_phrase_nodes"] = []

        parsed_response = {
            "text": input["text"],
            "idx": input.get("idx", 0),
            "gsw": gsw_dict,  # Always a dictionary now
            "context": input.get("context", ""),
            "doc_idx": input.get("doc_idx", input.get("idx", 0)),
            "global_id": input.get("global_id", "unknown"),
        }

        # Include sentence indices if available
        if "start_sentence" in input:
            parsed_response["start_sentence"] = input["start_sentence"]
        if "end_sentence" in input:
            parsed_response["end_sentence"] = input["end_sentence"]

        return parsed_response
