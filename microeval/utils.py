import re
from typing import List
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens.

    Uses NLTK sentence tokenization followed by regex word splitting.
    Falls back to pure regex if NLTK is unavailable.

    Args:
        text: Input string.

    Returns:
        List of lowercase token strings.
    """
    if text is None:
        return []
    text = text.strip()
    words = []
    try:
        sents = nltk.sent_tokenize(text)
        for s in sents:
            tokens = re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']", s)
            words.extend([t.lower() for t in tokens if t.strip()])
    except Exception:
        tokens = re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']", text)
        words = [t.lower() for t in tokens if t.strip()]
    return words


def ngrams(tokens, n):
    """Generate n-gram tuples from a token list.

    Args:
        tokens: List of token strings.
        n: N-gram size (must be >= 1).

    Returns:
        List of n-gram tuples.
    """
    if n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


def safe_div(a, b):
    """Safe division that returns 0.0 instead of raising ZeroDivisionError.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        a / b if b != 0, else 0.0.
    """
    return a / b if b else 0.0


def flatten(list_of_lists):
    """Flatten a list of lists into a single list.

    Args:
        list_of_lists: List of lists.

    Returns:
        Flat list containing all elements of all sublists.
    """
    return [x for lst in list_of_lists for x in lst]
