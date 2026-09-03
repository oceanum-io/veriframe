import os
from pathlib import Path
import pytest
from shapely.geometry import box

from rompy.core.source import SourceFile
from rompy.core.time import TimeRange

from veriframe.veriframe import VeriFrame
from veriframe.verisat import VeriSat


HERE = Path(__file__).parent
DATAMESH_TOKEN = os.getenv("DATAMESH_TOKEN")


@pytest.fixture(scope="module")
def source():
    return SourceFile(
        uri=HERE / "data/baltic.zarr",
        kwargs=dict(engine="zarr"),
    )


@pytest.fixture(scope="module")
def times():
    return TimeRange(
        start="20160101T00",
        end="20160201T00",
        freq="1h",
    )


def test_verisat_area(source):
    v1 = VeriSat(area=box(0, 0, 1, 1), model_source=source)
    v2 = VeriSat(area=(0, 0, 1, 1), model_source=source)
    assert v1 == v2


def test_load_model(source, times):
    v = VeriSat(
        area=(9, 53.8, 30.3, 66.0),
        model_source=source,
    )
    ds = v._load_model(times)
    t0, t1 = ds.time.to_index().to_pydatetime()[[0, -1]]
    assert (times.start >= t0) & (times.end <= t1)


@pytest.mark.skipif(not DATAMESH_TOKEN, reason="Datamesh token not in the environment")
def test_get_colocs(source, times):
    v = VeriSat(
        area=(9, 53.8, 30.3, 66.0),
        model_source=source,
        model_var="hs",
        offshore_buffer=1.0,
    )
    vf = v.get_colocs(times)
    assert isinstance(vf, VeriFrame)


def _fake_sat_dataset(times, lons, lats):
    """Altimeter query result in the shape datamesh currently returns."""
    import numpy as np
    import xarray as xr

    n = len(times)
    return xr.Dataset(
        {
            "platform": ("time", np.array(["JASON-3"] * n)),
            "swh_ku_cal": ("time", np.linspace(1.0, 3.0, n, dtype="float32")),
            "swh_ku_quality_control": ("time", np.array([1.0] * (n - 1) + [2.0], dtype="float32")),
        },
        coords={
            "time": times,
            "latitude": ("time", np.asarray(lats, dtype="float32")),
            "longitude": ("time", np.asarray(lons, dtype="float32")),
        },
    )


@pytest.fixture
def sat_query(source):
    """Satellite samples inside the baltic fixture, one repeated timestamp, one bad QC."""
    import numpy as np
    import pandas as pd

    ds = source.open()
    t0 = pd.Timestamp(ds.time.values[10])
    times = pd.to_datetime([t0, t0, t0 + pd.Timedelta("1h"), t0 + pd.Timedelta("2h"), t0 + pd.Timedelta("3h")])
    lons = np.linspace(float(ds.longitude[5]), float(ds.longitude[-5]), 5)
    lats = np.linspace(float(ds.latitude[5]), float(ds.latitude[-5]), 5)
    return times, lons, lats


def test_load_sat_dataset_and_dataframe(source, times, sat_query, monkeypatch):
    """_load_sat must accept the new time-indexed Dataset and the old DataFrame shape."""
    t, lons, lats = sat_query
    dset = _fake_sat_dataset(t, lons, lats)
    frame = dset.to_dataframe().reset_index()  # old shape: time as a column
    v = VeriSat(area=(9, 53.8, 30.3, 66.0), model_source=source, datamesh_token="x")
    for payload in (dset, frame):
        monkeypatch.setattr(type(v.datamesh), "query", lambda self, q: payload)
        df = v._load_sat(times)
        assert df.index.name == "time"
        assert df.index.is_monotonic_increasing
        assert {"latitude", "longitude", "platform", "swh_ku_cal"} <= set(df.columns)
        assert len(df) == 4  # the bad-QC sample is dropped, the duplicate time is kept


def test_get_colocs_with_dataset_query(source, times, sat_query, monkeypatch):
    t, lons, lats = sat_query
    v = VeriSat(area=(9, 53.8, 30.3, 66.0), model_source=source, model_var="hs", datamesh_token="x")
    monkeypatch.setattr(type(v.datamesh), "query", lambda self, q: _fake_sat_dataset(t, lons, lats))
    vf = v.get_colocs(times)
    assert isinstance(vf, VeriFrame)
    assert len(vf) == 4
    assert list(vf.columns[:5]) == ["lon", "lat", "platform", "satellite", "model"]
    assert vf.model.notna().any()


def test_load_sat_empty(source, times, monkeypatch):
    v = VeriSat(area=(9, 53.8, 30.3, 66.0), model_source=source, datamesh_token="x")
    monkeypatch.setattr(type(v.datamesh), "query", lambda self, q: None)
    assert v._load_sat(times) is None
