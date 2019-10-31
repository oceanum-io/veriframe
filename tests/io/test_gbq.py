from datetime import datetime
import pandas as pd
import pandas_gbq

import pytest

from onverify.io.gbq import dt_to_sql, sql_to_dt


def test_convert():
    seed = datetime(2000, 1, 1, 5, 10)
    timestamp = dt_to_sql(seed)
    print(seed)
    dt = sql_to_dt(timestamp)
    assert dt == seed


def test_insert():
    df = pd.DataFrame((dict(row1=[1, 2, 3], row2=[4, 5, 6], row3=[7, 8, 9])))
    pandas_gbq.to_gbq(
        df, "wave.test", project_id="oceanum-dev", if_exists="append"
    )
