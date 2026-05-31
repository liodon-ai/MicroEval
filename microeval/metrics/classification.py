from sklearn.metrics import f1_score, accuracy_score


def eval_classification(preds, refs):
    return {
        "accuracy": accuracy_score(refs, preds),
        "f1": f1_score(refs, preds, average="weighted"),
    }
