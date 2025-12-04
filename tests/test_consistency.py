import pandas as pd
import pytest

from case_base import CaseBase


def test_make_consistent(csv_file):
    CB = CaseBase(pd.read_csv(csv_file))
    initial_size = len(CB)
    CB.take_consistent_subset()
    reduced_size = len(CB)
    assert initial_size - reduced_size == 1


@pytest.mark.skip(reason="Slow test")
def test_make_consistent_mushroom():
    CB = CaseBase(pd.read_csv("data/mushroom.csv"))
    initial_size = len(CB)
    CB.take_consistent_subset()
    reduced_size = len(CB)
    assert initial_size - reduced_size == 26


@pytest.mark.skip(reason="Slow test")
def test_make_consistent_churn():
    CB = CaseBase(pd.read_csv("data/churn.csv"))
    initial_size = len(CB)
    CB.take_consistent_subset()
    reduced_size = len(CB)
    assert initial_size - reduced_size == 647


@pytest.mark.skip(reason="Slow test")
def test_make_consistent_admission():
    CB = CaseBase(pd.read_csv("data/admission.csv"))
    initial_size = len(CB)
    CB.take_consistent_subset()
    reduced_size = len(CB)
    assert initial_size - reduced_size == 16
