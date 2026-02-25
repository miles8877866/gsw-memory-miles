"""
Summary reranking using VoyageAI embeddings and cosine similarity.

This module implements the fourth step of the Q&A pipeline:
reranking entity summaries by relevance to the user's question.
"""

from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

try:
    from langchain_voyageai import VoyageAIEmbeddings
except ImportError:
    VoyageAIEmbeddings = None

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    GoogleGenerativeAIEmbeddings = None

class SummaryReranker:
    """
    Reranks entity summaries using VoyageAI embeddings and cosine similarity.
    
    This implements the paper's approach of using semantic similarity
    to rank summaries by relevance to the user's question, based on the
    rerank_summaries function from the original gsw_qa.py.
    """
    
    def __init__(self, embedding_model_name: str = "voyage-3"):
        """
        Initialize the summary reranker.
        
        Args:
            embedding_model_name: Name of embedding model to use. 
                                Supports 'voyage-*' and 'google_genai/*' or 'models/*' for Gemini.
        """
        self.embedding_model_name = embedding_model_name
        
    def _get_embedding_model(self):
        """Helper to initialize the correct embedding model based on name."""
        if "voyage" in self.embedding_model_name.lower():
            if VoyageAIEmbeddings is None:
                raise ImportError("langchain-voyageai not installed")
            return VoyageAIEmbeddings(model=self.embedding_model_name)
        elif "google" in self.embedding_model_name.lower() or "gemini" in self.embedding_model_name.lower() or "models/" in self.embedding_model_name.lower():
            if GoogleGenerativeAIEmbeddings is None:
                # Try to import it here in case it was installed later
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                except ImportError:
                    raise ImportError("langchain-google-genai not installed. Please install it to use Gemini embeddings.")
            
            # Extract model name if it's in google_genai/model-name format
            model_name = self.embedding_model_name
            if "/" in model_name:
                model_name = model_name.split("/")[-1]
            
            return GoogleGenerativeAIEmbeddings(model=model_name)
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model_name}")
        
    def rerank_summaries(
        self,
        summaries: List[Tuple[str, str]],  # (entity_name, summary)
        question: str,
        max_summaries: int = 17
    ) -> List[Tuple[str, str, float]]:  # (entity_name, summary, score)
        """
        Rerank summaries by semantic similarity to the question.
        
        This replicates the rerank_summaries function from the original
        gsw_qa.py (lines 671-752) with simplified error handling.
        
        Args:
            summaries: List of (entity_name, summary) tuples
            question: The user's question
            max_summaries: Maximum number of summaries to return
            
        Returns:
            List of (entity_name, summary, similarity_score) tuples,
            ranked by decreasing similarity score
        """
        if not summaries:
            print("No relevant summaries provided for reranking.")
            return []

        # Filter out empty summaries
        valid_summaries_data = []
        summary_texts_to_embed = []
        for entity_name, summary_text in summaries:
            if summary_text and summary_text.strip():
                valid_summaries_data.append((entity_name, summary_text))
                summary_texts_to_embed.append(summary_text)

        if not summary_texts_to_embed:
            print("All summaries were empty or None; nothing to rerank.")
            return []

        # Initialize embedding model
        embedding_model = self._get_embedding_model()

        # Get question embedding
        question_embedding = embedding_model.embed_query(question)
        question_embedding_np = np.array(question_embedding).reshape(1, -1)

        # Get summary embeddings
        summary_embeddings_list = embedding_model.embed_documents(summary_texts_to_embed)
        summary_embeddings_np = np.array(summary_embeddings_list)

        # Handle edge cases for embeddings shape
        if summary_embeddings_np.ndim == 1:
            if summary_embeddings_np.size > 0 and len(summary_texts_to_embed) == 1:
                summary_embeddings_np = summary_embeddings_np.reshape(1, -1)
            else:
                print(
                    f"Warning: Unexpected summary embeddings shape: {summary_embeddings_np.shape}. Returning original valid summaries."
                )
                return [(name, summary, 0.0) for name, summary in valid_summaries_data[:max_summaries]]

        if summary_embeddings_np.shape[0] == 0:
            print("No summary embeddings were generated.")
            return [(name, summary, 0.0) for name, summary in valid_summaries_data[:max_summaries]]

        if summary_embeddings_np.shape[0] != len(summary_texts_to_embed):
            print(
                f"Mismatch between number of summaries ({len(summary_texts_to_embed)}) and embeddings ({summary_embeddings_np.shape[0]}). Returning original valid summaries."
            )
            return [(name, summary, 0.0) for name, summary in valid_summaries_data[:max_summaries]]

        # Calculate cosine similarities
        similarities = cosine_similarity(
            question_embedding_np, summary_embeddings_np
        ).flatten()

        # Create scored summaries
        scored_summaries = []
        for i, (entity_name, summary_text) in enumerate(valid_summaries_data):
            scored_summaries.append({
                "entity_name": entity_name,
                "summary": summary_text, 
                "score": similarities[i]
            })

        # Sort by score (descending) and take top k
        reranked_scored_summaries = sorted(
            scored_summaries, key=lambda x: x["score"], reverse=True
        )
        
        # Return as list of tuples with scores
        final_reranked_summaries = [
            (item["entity_name"], item["summary"], item["score"]) 
            for item in reranked_scored_summaries[:max_summaries]
        ]

        return final_reranked_summaries

