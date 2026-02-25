"""
Entity extraction from questions using LLM-based NER.

This module implements the first step of the Q&A pipeline:
extracting named entities from user questions to serve as entry points into the GSW.
"""

import os
from typing import List

from bespokelabs import curator


class QuestionEntityExtractor(curator.LLM):
    """
    Extracts named entities from questions using LLM-based NER.

    This follows the paper's approach and replicates the QuestionEntityExtractor
    from the original gsw_qa.py, using curator for efficient batched processing.
    """

    return_completions_object = True
    require_all_responses = False

    def __init__(self, **kwargs):
        # Pop parameters that curator doesn't like in __init__
        kwargs.pop("max_concurrent_requests", None)
        kwargs.pop("require_all_responses", None)
        
        super().__init__(**kwargs)
        self.require_all_responses = False

    def prompt(self, input_data):
        """Create prompt for entity extraction following original pattern."""
        system_prompt = (
            """You are a helpful assistant that extracts named entities from text."""
        )

        user_prompt = f"""Identify and extract all named entities (people, places, objects, organizations, etc.) from the following question.

                Note that for dates, you should extract the year, month and day as the SAME entity.

            For example:
                        "April 15, 2024" should be extracted as ONE entity "April 15, 2024" NOT as two entities "April" and "15, 2024"

                        Question: "{input_data["question"]}"

                        Return only the entities as a pipe-delimited list (using | as separator), with no explanations or additional text."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input_data, response):
        """Parse LLM response to extract entities."""
        entities_text = response["choices"][0]["message"]["content"].strip()
        entities = [entity.strip() for entity in entities_text.split("|")]
        return {
            "question": input_data["question"],
            "entities": entities,
        }

    def extract_entities(self, question: str) -> List[str]:
        """
        Extract named entities from a single question.

        Args:
            question: The user's question

        Returns:
            List of extracted entity names
        """
        results = self.extract_entities_batch([question])
        return results[0]

    def extract_entities_batch(self, questions: List[str]) -> List[List[str]]:
        """
        Extract named entities from multiple questions efficiently.

        Args:
            questions: List of questions

        Returns:
            List of entity lists, one per question
        """
        # Format questions for curator
        question_inputs = [{"question": q} for q in questions]

        # Call curator with batch (Sequential fallback for Windows/Robustness)
        if os.name == 'nt':
            print(f"--- Extracting entities from {len(question_inputs)} questions sequentially ---")
            from tqdm import tqdm
            results_list = []
            for inp in tqdm(question_inputs, desc="Extracting Entities"):
                try:
                    res = self([inp])
                    results_list.extend(res.dataset)
                except Exception as e:
                    print(f"Warning: Entity extraction failed for question: {inp.get('question')[:50]}... Error: {e}")
            
            # Mock the curator Response object structure
            class MockResponse:
                def __init__(self, dataset):
                    self.dataset = dataset
            results = MockResponse(results_list)
        else:
            results = self(question_inputs)

        # Extract just the entities from results using .dataset
        return [result["entities"] for result in results.dataset]

