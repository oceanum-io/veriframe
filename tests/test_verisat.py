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
