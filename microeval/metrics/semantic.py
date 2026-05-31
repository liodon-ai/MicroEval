import warnings

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

_encoder = None


def _load_encoder(model_name="all-MiniLM-L6-v2"):
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
    encoder = _load_encoder(model_name)
    pred_embs = encoder.encode(preds)
    ref_embs = encoder.encode(refs)
    similarities = cosine_similarity(pred_embs, ref_embs).diagonal()
    return {"bert_score": float(np.mean(similarities))}


def semantic_similarity(preds, refs, model_name="all-MiniLM-L6-v2"):
    return bert_score(preds, refs, model_name=model_name)
