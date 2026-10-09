"""Beki's feature: precision@k. Remove the skip once metrics.precision_at_k is written."""
import pytest

from rag_eval import metrics as M
from rag_eval.corpus import Chunk

R = Chunk("a#s#0", "a", "s", "the answer is 42")
N = Chunk("b#s#0", "b", "s", "unrelated text")
ITEM = {"doc": "a", "evidence": ["42"]}


@pytest.mark.skip(reason="Beki writes metrics.precision_at_k without AI")
def test_precision_at_k():
    assert M.precision_at_k([R, N, R], ITEM, 3) == pytest.approx(2 / 3)
    assert M.precision_at_k([N, N, R], ITEM, 1) == 0
    assert M.precision_at_k([R], ITEM, 1) == 1
