import pandas as pd
import pytest

from case_base import CaseBase
from precedents import get_best_precedents


@pytest.fixture()
def inconst_csv(tmp_path_factory):
    data = {
        "Gift": [1, 1, 1, 1, 1, 1],
        "Present": [1, 1, 1, 1, 1, 1],
        "Website": [0, 0, 0, 2, 2, 2],
        "High-cost": [0, 0, 0, 0, 0, 0],
        "Label": [1, 0, 1, 1, 1, 1],
    }
    df = pd.DataFrame(data)
    path = tmp_path_factory.mktemp("data") / "inconst.csv"
    df.to_csv(path, index=False, header=True)
    return path


def test_find_best_precedents_naive(inconst_csv):
    CB = CaseBase(pd.read_csv(inconst_csv))
    case = CB[0]
    precedents = get_best_precedents(case, CB)
    assert len(precedents) == 1
    assert precedents[0]["name"] == 2


def test_find_best_precedents_rel(inconst_csv):
    CB = CaseBase(pd.read_csv(inconst_csv), auth_method="relative")
    case = CB[0]
    precedents = get_best_precedents(case, CB)
    assert len(precedents) == 4
    assert precedents[0]["name"] == 2
    assert precedents[1]["name"] == 3
    assert precedents[2]["name"] == 4
    assert precedents[3]["name"] == 5


def test_find_best_precedents_abs(inconst_csv):
    CB = CaseBase(pd.read_csv(inconst_csv), auth_method="absolute")
    case = CB[0]
    precedents = get_best_precedents(case, CB)
    assert len(precedents) == 1
    assert precedents[0]["name"] == 2
