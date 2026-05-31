import numpy as np
import warnings

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

_encoder = None


def _load_encoder(model_name="all-MiniLM-L6-v2"):
    """Load and cache a SentenceTransformer encoder."""
    global _encoder
    if _encoder is None:
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for semantic similarity. "
                "Install: pip install microeval[semantic]"
            )
        _encoder = SentenceTransformer(model_name)
    return _encoder


def bert_score(preds, refs, model_name="all-MiniLM-L6-v2"):
    """Compute semantic similarity between predictions and references using sentence embeddings.

    Args:
        preds: List of predicted strings.
        refs: List of reference strings.
        model_name: SentenceTransformer model name (default: all-MiniLM-L6-v2).

    Returns:
        dict with key "bert_score" (float, 0-1, higher = more semantically similar).
    """
    encoder = _load_encoder(model_name)
    pred_embs = encoder.encode(preds)
    ref_embs = encoder.encode(refs)
    similarities = cosine_similarity(pred_embs, ref_embs).diagonal()
    return {"bert_score": float(np.mean(similarities))}


def semantic_similarity(preds, refs, model_name="all-MiniLM-L6-v2"):
    """Alias for bert_score. Returns embedding cosine similarity."""
    return bert_score(preds, refs, model_name=model_name)
