
import google.auth
from google.cloud import bigquery


sql = """
    SELECT time, swh
    FROM `oceanum-prod.cersat.data`
    ORDER by time desc
    LIMIT 5
"""


client = bigquery.Client()

# Run a Standard SQL query using the environment's default project
df = client.query(sql).to_dataframe()

# Run a Standard SQL query with the project set explicitly
project_id = 'oceanum-dev'
df = client.query(sql, project=project_id).to_dataframe()
