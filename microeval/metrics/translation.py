import sacrebleu


def eval_translation(preds, refs):
    """Evaluate machine translation outputs using corpus-level BLEU.

    Args:
        preds: List of translated strings.
        refs: List of reference translation strings (single reference per hypothesis).

    Returns:
        dict with key "bleu" (float, 0-100, higher is better).
    """
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return {"bleu": bleu.score}
