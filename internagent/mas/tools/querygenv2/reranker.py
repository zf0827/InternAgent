"""
Reranker Module for V2

Implements two-stage reranking using BGE models:
1. First stage: Fast embedding-based retrieval using bge-base-en-v1.5
2. Second stage: Precise reranking using bge-reranker-base
"""

import logging
import os
import numpy as np
from typing import List, Tuple, Optional

# Configure Hugging Face mirror endpoint if not set
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer, CrossEncoder, util

logger = logging.getLogger(__name__)


def rerank_articles_two_stage(
    core_article: str,
    article_list: List[str],
    top_k: int = 20,
    embedding_model_name: str = "BAAI/bge-base-en-v1.5",
    reranker_model_name: str = "BAAI/bge-reranker-base"
) -> List[Tuple[str, float, float]]:
    """
    Two-stage article reranking function.
    
    Stage 1 (Embedding Retrieval):
    - Uses bge-base-en-v1.5 model to generate embeddings for the core article and all articles
    - Adds query instruction prefix to the core article to improve retrieval effectiveness
    - Computes cosine similarity and filters top-k candidate articles
    
    Stage 2 (Reranker Reranking):
    - Uses bge-reranker-base model to precisely score candidate articles
    - Reranks candidate articles based on reranker scores
    
    Query Design Notes:
    - Embedding stage: Query (core article) adds instruction prefix "Represent this sentence for searching relevant passages:"
                      Documents (article collection) use original text without prefix
    - Reranker stage: Both query and documents use full text without instruction prefix
    
    Args:
        core_article: Core article content (used as query)
        article_list: List of articles to be ranked
        top_k: Number of candidate articles to filter in stage 1 (default: 20)
        embedding_model_name: Embedding model name
        reranker_model_name: Reranker model name
    
    Returns:
        Ranked article list, each element is (article_content, embedding_similarity_score, reranker_score)
    """
    
    if not article_list:
        logger.warning("Empty article list provided for reranking")
        return []
    
    logger.info(f"Loading models: {embedding_model_name} and {reranker_model_name}")
    
    # Load models
    # Embedding model: for fast retrieval
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Reranker model: for precise reranking
    reranker_model = CrossEncoder(reranker_model_name)
    
    logger.info(f"Stage 1: Embedding retrieval (selecting top-{top_k} from {len(article_list)} articles)")
    
    # ========== Stage 1: Embedding Retrieval ==========
    
    # Add query instruction prefix to core article (improves retrieval effectiveness)
    # According to BGE documentation, v1.5 can work without prefix, but adding it improves results
    query_text = f"Represent this sentence for searching relevant passages: {core_article}"
    
    # Batch encoding: query and documents
    # Query uses text with instruction prefix, documents use original text
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=True)
    article_embeddings = embedding_model.encode(article_list, convert_to_tensor=True)
    
    # Compute cosine similarity
    # query_embedding shape: [768]
    # article_embeddings shape: [n_articles, 768]
    # cosine_scores shape: [n_articles]
    cosine_scores = util.cos_sim(query_embedding, article_embeddings)[0]
    
    # Get top-k candidate article indices and scores
    # If number of articles is less than top_k, use all articles
    actual_top_k = min(top_k, len(article_list))
    top_k_indices = np.argsort(cosine_scores.cpu().numpy())[-actual_top_k:][::-1]
    top_k_articles = [article_list[i] for i in top_k_indices]
    top_k_scores = [float(cosine_scores[i]) for i in top_k_indices]
    
    logger.info(f"Selected {len(top_k_articles)} candidate articles")
    if top_k_scores:
        logger.info(f"Similarity score range: {min(top_k_scores):.4f} - {max(top_k_scores):.4f}")
    
    logger.info(f"Stage 2: Reranker precise reranking (scoring {len(top_k_articles)} candidate articles)")
    
    # ========== Stage 2: Reranker Reranking ==========
    
    # Build query-document pair list
    # Reranker model input format: [query_text, document_text]
    # Note: reranker stage does not need instruction prefix, use original text directly
    pairs = [[core_article, article] for article in top_k_articles]
    
    # Batch scoring (reranker model automatically handles batch input)
    reranker_scores = reranker_model.predict(pairs)
    
    # Convert reranker scores to list (if numpy array)
    if isinstance(reranker_scores, np.ndarray):
        reranker_scores = reranker_scores.tolist()
    elif not isinstance(reranker_scores, list):
        reranker_scores = list(reranker_scores)
    
    # Combine results: (article_content, embedding_score, reranker_score)
    results = list(zip(top_k_articles, top_k_scores, reranker_scores))
    
    # Sort by reranker score in descending order (higher score = higher relevance)
    results.sort(key=lambda x: x[2], reverse=True)
    
    if reranker_scores:
        logger.info(f"Reranking completed, reranker score range: {min(reranker_scores):.4f} - {max(reranker_scores):.4f}")
    
    return results

