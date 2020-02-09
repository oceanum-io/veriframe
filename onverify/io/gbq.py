import google.auth
from pandas import read_gbq
from datetime import datetime

sqltimefmt = "%Y-%m-%d %H:%M:%S"


def dt_to_sql(dt):
    return dt.strftime(sqltimefmt)


def sql_to_dt(sql):
    return datetime.strptime(sql, sqltimefmt)


def retrieve(sql, project_id="oceanum-dev", use_bqstorage_api=False):
    """
    Retrieve data from bigquery database and return as pandas dataframe
    Args:
        sql (str):          bigquery query string
        project_id (str):   project_id
    """

    df = read_gbq(sql, project_id=project_id, use_bqstorage_api=use_bqstorage_api)
    return df


class GBQAlt:
    """Class for performing altimeter data retrievals"""

    def __init__(
        self,
        variables=["time", "swh", "lat", "lon"],
        dset="oceanum-prod.cersat.data",
        project_id="oceanum-prod",
        use_bqstorage_api=False,
    ):
        self.variables = variables
        self.dset = dset
        self.project_id = project_id
        self.use_bqstorage_api = use_bqstorage_api

    def contruct_sql(self, start, end):
        if not start:
            sql = """
                SELECT {}
                FROM `{}`
                ORDER by time asc
            """.format(
                ", ".join(self.variables), self.dset
            )
        else:
            sql = """
                SELECT {}
                FROM `{}`
                WHERE time BETWEEN "{}" AND "{}"
                ORDER by time asc
            """.format(
                ", ".join(self.variables), self.dset, dt_to_sql(start), dt_to_sql(end)
            )
        return sql

    def get(self, start=None, end=None):
        sql = self.contruct_sql(start, end)
        self.df = retrieve(sql, project_id=self.project_id, use_bqstorage_api=self.use_bqstorage_api)
