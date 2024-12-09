import pytest
from shapely.geometry import box

from rompy.core.source import SourceFile
from rompy.core.time import TimeRange

from veriframe.satellite import VeriSat


@pytest.fixture(scope="module")
def source():
    return SourceFile(
        uri="/source/veriframe/tests/data/baltic.zarr",
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
    v1 = VeriSat(area=box(0, 0, 1, 1), model_souce=source)
    v2 = VeriSat(area=(0, 0, 1, 1), model_souce=source)
    assert v1 == v2


def test_load_model(source, times):
    v = VeriSat(
        area=(9, 53.8, 30.3, 66.0),
        model_source=source,
    )
    ds = v._load_model(times)
    t0, t1 = ds.time.to_index().to_pydatetime()[[0, -1]]
    assert (times.start >= t0) & (times.end <= t1)


def test_run(source, times):
    v = VeriSat(
        area=(9, 53.8, 30.3, 66.0),
        model_source=source,
        model_var="hs",
    )
    data = v.run(times)
    import ipdb; ipdb.set_trace()
    # assert v.model_type == "file"
    # assert v.uri == "tests/data/verisat.nc"
