import pytest
from microeval.utils import tokenize, ngrams, safe_div, flatten


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Hello world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty(self):
        assert tokenize("") == []

    def test_none(self):
        assert tokenize(None) == []

    def test_lowercase(self):
        tokens = tokenize("HELLO")
        assert tokens == ["hello"]

    def test_punctuation(self):
        tokens = tokenize("Hello, world!")
        assert "hello" in tokens
        assert "world" in tokens


class TestNgrams:
    def test_unigrams(self):
        result = ngrams(["a", "b", "c"], 1)
        assert result == [("a",), ("b",), ("c",)]

    def test_bigrams(self):
        result = ngrams(["a", "b", "c"], 2)
        assert result == [("a", "b"), ("b", "c")]

    def test_n_too_large(self):
        assert ngrams(["a", "b"], 5) == []

    def test_n_zero(self):
        assert ngrams(["a"], 0) == []

    def test_n_negative(self):
        assert ngrams(["a"], -1) == []


class TestSafeDiv:
    def test_normal(self):
        assert safe_div(10, 2) == 5.0

    def test_zero_denominator(self):
        assert safe_div(10, 0) == 0.0

    def test_negative(self):
        assert safe_div(-10, 2) == -5.0


class TestFlatten:
    def test_basic(self):
        assert flatten([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]

    def test_empty(self):
        assert flatten([]) == []

    def test_nested_empty(self):
        assert flatten([[], []]) == []
