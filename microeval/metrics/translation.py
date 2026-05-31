import sacrebleu


def eval_translation(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return {"bleu": bleu.score}
