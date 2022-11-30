import pandas as pd

from case_base import CaseBase
from precedents import get_precedent_distribution


def test_best_precedent_distribution_naive(csv_file):
    CB = CaseBase(pd.read_csv(csv_file))
    dict = get_precedent_distribution(CB)
    assert round(dict["mean"], 2) == 2.71
    assert round(dict["std"], 2) == 1.58


def test_best_precedent_distribution_rel(csv_file):
    CB = CaseBase(pd.read_csv(csv_file), auth_method="relative")
    dict = get_precedent_distribution(CB)
    assert round(dict["mean"], 2) == 3.43
    assert round(dict["std"], 2) == 1.68
