# MicroEval

MicroEval is a tiny, dependency-light toolkit to quickly evaluate NLP/LLM outputs. It's designed for fast smoke-tests and demos — not to replace heavyweight benchmarks.

## Features
- Simple n-gram perplexity
- ROUGE-1, ROUGE-2, ROUGE-L (basic)
- BLEU-like approximation
- Flesch Reading Ease
- Tiny lexicon-based toxicity score
- CLI for quick evaluations

## Install
```bash
pip install numpy nltk
# optional grammar checks:
# pip install language-tool-python
```
