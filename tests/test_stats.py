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
        ("usi", 0.114, {}),
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



def test_usi_is_blind_to_offset_and_gain(data):
    """USI must not move under an affine transform of the model.

    This is the property that distinguishes it from SI: a model that is
    perfectly correlated but mis-scaled has real skill in timing and none in
    amplitude, and USI is the index that says so.
    """
    import numpy as np

    obs = data["hs_obs"]
    for alpha, beta in [(1.0, 0.0), (1.0, 0.5), (1.3, 0.0), (0.7, -0.2)]:
        df = data.assign(model=alpha * data["hs_hds"] + beta)
        vf = VeriFrame(df, ref_col="hs_obs", verify_col="model")
        ref = VeriFrame(data, ref_col="hs_obs", verify_col="hs_hds")
        assert vf.usi() == pytest.approx(ref.usi(), rel=1e-9)

    # ... whereas a pure gain error is invisible to USI and not to SI.
    perfect = data.assign(model=1.3 * obs)
    vf = VeriFrame(perfect, ref_col="hs_obs", verify_col="model")
    assert vf.usi() == pytest.approx(0.0, abs=1e-9)
    assert vf.si() > 0.1


def test_mse_decomposition_is_exact(data):
    """The three terms must sum to the MSE."""
    import numpy as np

    vf = VeriFrame(data, ref_col="hs_obs", verify_col="hs_hds")
    d = vf.mse_decomposition()
    total = d["bias2"] + d["amplitude"] + d["decorrelation"]
    assert total == pytest.approx(d["mse"], rel=1e-12)
    assert sum(d[f"{k}_share"] for k in
               ("bias2", "amplitude", "decorrelation")) == pytest.approx(1.0)
