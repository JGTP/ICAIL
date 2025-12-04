import pandas as pd
import pytest

from case_base import CaseBase


@pytest.fixture
def simple_csv(tmp_path_factory):
    data = {
        "Col1": [1, 1, 1, 0],
        "Label": [1, 1, 0, 0],
    }
    df = pd.DataFrame(data)
    path = tmp_path_factory.mktemp("data") / "simple.csv"
    df.to_csv(path, index=False, header=True)
    return path


def test_number_of_inconsistent_forcing_relations_naive(simple_csv):
    CB = CaseBase(pd.read_csv(simple_csv))
    inds = range(len(CB))
    forcings = CB.get_forcings(inds)
    assert len(forcings) == 11
    Id = CB.determine_inconsistent_forcings(inds, forcings)
    n_inconst_forcings = CB.get_n_inconst_forcings(Id)
    assert n_inconst_forcings == 4


def test_number_of_inconsistent_forcing_relations_relative(simple_csv):
    CB = CaseBase(pd.read_csv(simple_csv), auth_method="relative")
    inds = range(len(CB))
    forcings = CB.get_forcings(inds)
    assert len(forcings) == 9
    Id = CB.determine_inconsistent_forcings(inds, forcings)
    n_inconst_forcings = CB.get_n_inconst_forcings(Id)
    assert n_inconst_forcings == 0


def test_number_of_inconsistent_forcing_admission():
    CB = CaseBase(pd.read_csv("data/admission.csv"), auth_method="relative")
    inds = range(len(CB))
    forcings = CB.get_forcings(inds)
    assert len(forcings) == 53202
    Id = CB.determine_inconsistent_forcings(inds, forcings)
    n_inconst_forcings = CB.get_n_inconst_forcings(Id)
    assert n_inconst_forcings == 0
