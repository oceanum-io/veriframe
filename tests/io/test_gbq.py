from datetime import datetime

import pytest

from onverify.io.gbq import dt_to_sql, sql_to_dt


def test_convert():
    seed = datetime(2000, 1, 1, 5, 10)
    timestamp = dt_to_sql(seed)
    print(seed)
    dt = sql_to_dt(timestamp)
    assert dt == seed

