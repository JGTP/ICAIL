import pandas as pd
import pytest

from authoritativeness import relative_authoritativeness
from case_base import CaseBase


@pytest.fixture
def simple_csv(tmp_path_factory):
    data = {
        "Col1": [0, 0, 1, 1, 2, 2],
        "Label": [0, 0, 1, 1, 1, 0],
    }
    df = pd.DataFrame(data)
    path = tmp_path_factory.mktemp("data") / "simple.csv"
    df.to_csv(path, index=False, header=True)
    return path


def test_comparison_naive(simple_csv):
    CB = CaseBase(pd.read_csv(simple_csv))
    assert CB[3] <= CB[4]


def test_comparison_relative(simple_csv):
    CB = CaseBase(pd.read_csv(simple_csv))
    assert (CB[3] <= CB[4]) and not (
        relative_authoritativeness(CB[3], CB) <= relative_authoritativeness(CB[4], CB)
    )
