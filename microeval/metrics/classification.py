from sklearn.metrics import f1_score, accuracy_score


def eval_classification(preds, refs):
    """Evaluate classification predictions against ground truth labels.

    Args:
        preds: List of predicted label strings.
        refs: List of ground truth label strings.

    Returns:
        dict with keys "accuracy" (float, 0-1) and "f1" (float, 0-1, weighted).
    """
    return {
        "accuracy": accuracy_score(refs, preds),
        "f1": f1_score(refs, preds, average="weighted"),
    }
