from rouge_score import rouge_scorer


def eval_summarization(preds, refs):
    """Evaluate summarization outputs using ROUGE scores.

    Computes ROUGE-1, ROUGE-2, and ROUGE-L F-measures with stemming,
    averaged across all prediction-reference pairs.

    Args:
        preds: List of generated summary strings.
        refs: List of reference summary strings.

    Returns:
        dict with keys "rouge1", "rouge2", "rougeL" (float, 0-1 each).
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p) for r, p in zip(refs, preds)]
    avg_scores = {
        k: sum(s[k].fmeasure for s in scores) / len(scores) for k in scores[0]
    }
    return avg_scores
