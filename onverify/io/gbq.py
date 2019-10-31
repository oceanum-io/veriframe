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

    def contruct_sql(self, start, end):
        sql = """
            SELECT {}
            FROM `{}`
            WHERE time BETWEEN "{}" AND "{}"
            ORDER by time asc
        """.format(
            ", ".join(self.variables), self.dset, dt_to_sql(start), dt_to_sql(end)
        )
        return sql

    def get(self, start, end):
        sql = self.contruct_sql(start, end)
        self.df = retrieve(sql, project_id=self.project_id)


