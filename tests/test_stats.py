from pathlib import Path
import pytest
import pandas as pd

from veriframe.veriframe import VeriFrame


DATADIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def data():
    """Colocs dataframe to use with tests."""
    df = pd.read_csv(DATADIR / "colocs.csv")
    df.index = pd.to_datetime(df["time"])
    yield df.drop("time", axis=1)


@pytest.mark.parametrize(
    "stat, value, kwargs",
    [
        ("bias", 0.088, {}),
        ("bias", 0.041, dict(norm=True)),
        ("rmsd", 0.261, {}),
        ("rmsd", 0.122, dict(norm=True)),
        ("si", 0.114, {}),
        ("mad", 0.204, {}),
        ("mad", 0.095, dict(norm=True)),
        ("mrad", 0.105, {}),
        ("ks", 0.087, {}),
    ]
)
def test_stats(data, stat, value, kwargs):
    """Test stats methods from VeriFrame."""
    vf = VeriFrame(data, ref_col="hs_obs", verify_col="hs_hds")
    assert getattr(vf, stat)(**kwargs) == pytest.approx(value, rel=0.01)

