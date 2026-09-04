"""Tests for the Taylor diagram."""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from veriframe.taylor import TaylorDiagram, df2taylor

DATADIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(DATADIR / "colocs.csv")
    df.index = pd.to_datetime(df["time"])
    return df.drop("time", axis=1)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_default_columns_skip_reference_coords_and_text(data):
    """A coloc frame carries lon/lat/site columns; they are not models."""
    df = pd.DataFrame(
        {"obs": data["hs_obs"], "hds": data["hs_hds"], "lon": data["lon_hds"],
         "lat": data["lat_hds"], "site": "buoy"}
    )
    dia = df2taylor(df, obslabel="obs")
    labels = [p.get_label() for p in dia.samplePoints]
    assert labels == ["Reference", "hds"]


def test_draws_on_given_diagram_and_its_figure(data, tmp_path):
    """With ``dia`` given, nothing may go to a fresh figure."""
    fig = plt.figure()
    dia = TaylorDiagram(data["hs_obs"].std(ddof=1), 3.0, fig=fig, label="ref")
    out = df2taylor(data, obslabel="hs_obs", mod_cols=["hs_hds"], dia=dia,
                    label="ref", plotdir=str(tmp_path))
    assert out is dia
    assert dia._ax.get_legend() is not None
    assert (tmp_path / "ref.png").exists()


def test_legend_is_on_the_diagram_axes_not_the_figure(data):
    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    dia = df2taylor(data, obslabel="hs_obs", mod_cols=["hs_hds"], fig=fig,
                    rect=224)
    assert dia._ax.figure is fig
    assert dia._ax.get_legend() is not None
    assert fig.legends == []


def test_labels_and_colors(data):
    dia = df2taylor(data, obslabel="hs_obs", mod_cols=["hs_hds"],
                    mod_labels=["Hindcast"], colors=["red"])
    (sample,) = dia.samplePoints[1:]
    assert sample.get_label() == "Hindcast"
    assert sample.get_color() == "red"


def test_negative_correlation_and_wide_spread_are_drawn(data):
    """Neither a negative R nor a spread beyond 1.5x the reference is lost."""
    obs = data["hs_obs"]
    df = pd.DataFrame({"obs": obs, "anti": -obs, "wide": 3.0 * obs})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dia = df2taylor(df, obslabel="obs")
    assert dia.negative
    assert dia.smax > 3.0 * obs.std(ddof=1)
    assert len(dia.samplePoints) == 3


def test_add_sample_warns_when_sample_cannot_be_shown():
    dia = TaylorDiagram(1.0, 1.5)
    with pytest.warns(UserWarning, match="negative=True"):
        dia.add_sample(1.0, -0.5, label="anti")
    with pytest.warns(UserWarning, match="maxstd"):
        dia.add_sample(2.0, 0.9, label="wide")
    with pytest.warns(UserWarning, match="NaN"):
        assert dia.add_sample(np.nan, 0.9, label="nan") is None


def test_many_models_get_distinct_colours(data):
    obs = data["hs_obs"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"obs": obs})
    for i in range(9):
        df[f"m{i}"] = obs + rng.normal(0, 0.1 * (i + 1), len(obs))
    dia = df2taylor(df, obslabel="obs")
    colours = {tuple(np.atleast_1d(p.get_color())) for p in dia.samplePoints[1:]}
    assert len(colours) == 9
