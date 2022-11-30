import pandas as pd

from authoritativeness import (absolute_authoritativeness,
                               harmonic_authoritativeness,
                               product_authoritativeness,
                               relative_authoritativeness)
from case_base import CaseBase


def test_absolute_authoritativeness(csv_file):
    CB = CaseBase(pd.read_csv(csv_file))
    case = CB[0]
    auth = absolute_authoritativeness(case, CB)
    assert auth == 6 / 7


def test_relative_authoritativeness(csv_file):
    CB = CaseBase(pd.read_csv(csv_file))
    case = CB[5]
    auth = relative_authoritativeness(case, CB)
    assert auth == 1 / 6


def test_product_authoritativeness(csv_file):
    CB = CaseBase(pd.read_csv(csv_file))
    case = CB[0]
    auth = product_authoritativeness(case, CB)
    assert auth == (6 / 7) * (6 / 7)


def test_harmonic_authoritativeness(csv_file):
    CB = CaseBase(pd.read_csv(csv_file))
    case = CB[0]
    auth = harmonic_authoritativeness(case, CB, beta=1)
    assert auth == 2 * ((6 / 7) * (6 / 7)) / ((6 / 7) + (6 / 7))
