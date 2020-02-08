import google.auth
from google.cloud import bigquery
from datetime import datetime

sqltimefmt = "%Y-%m-%d %H:%M:%S"


def dt_to_sql(dt):
    return dt.strftime(sqltimefmt)


def sql_to_dt(sql):
    return datetime.strptime(sql, sqltimefmt)


def retrieve(sql, project_id="oceanum-dev"):
    """
    Retrieve data from bigquery database and return as pandas dataframe
    Args:
        sql (str):          bigquery query string
        project_id (str):   project_id
    """

    client = bigquery.Client()
    df = client.query(sql, project=project_id).to_dataframe()
    return df


class GBQAlt:
    """Class for performing altimeter data retrievals"""

    def __init__(
        self,
        variables=["time", "swh", "lat", "lon"],
        dset="oceanum-prod.cersat.data",
        project_id="oceanum-prod",
    ):
        self.variables = variables
        self.dset = dset
        self.project_id = project_id

    def contruct_sql(self, start, end, x0, x1, y0, y1):
        sql = f"SELECT {','.join(self.variables)} FROM `{self.dset}`"
        clause = "WHERE"
        if start:
            sql += f" {clause} time BETWEEN '{dt_to_sql(start)}' AND '{dt_to_sql(end)}'"
            clause = "AND"
        if x0:
            sql += f" {clause} lon BETWEEN {x0} AND {x1} AND lat BETWEEN {y0} AND {y1}"
        sql += " ORDER BY time ASC"
        return sql

    def get(self, start=None, end=None, x0=None, x1=None, y0=None, y1=None):
        sql = self.contruct_sql(start, end, x0, x1, y0, y1)
        self.df = retrieve(sql, project_id=self.project_id)
        return self.df
