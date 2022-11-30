import pandas as pd
import pytest


@pytest.fixture(scope="session")
def csv_file(tmp_path_factory):
    data = {
        "Gift": [1, 1, 1, 1, 1, 1, 1],
        "Present": [1, 1, 1, 1, 1, 1, 1],
        "Website": [0, 0, 0, 2, 2, 2, 15],
        "High-cost": [0, 0, 0, 0, 0, 0, 0],
        "Label": [1, 1, 1, 1, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    path = tmp_path_factory.mktemp("data") / "test.csv"
    df.to_csv(path, index=False, header=True)
    return path
