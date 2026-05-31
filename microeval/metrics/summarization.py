from rouge_score import rouge_scorer


def eval_summarization(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p) for r, p in zip(refs, preds)]
    avg_scores = {
        k: sum(s[k].fmeasure for s in scores) / len(scores) for k in scores[0]
    }
    return avg_scores
