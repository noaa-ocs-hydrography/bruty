import psycopg2
import pickle
raise Exception("supply connection parameters")
URL_FILENAME = r""
CREDENTIALS_FILENAME = r""

url = 'OCS-VS-NBS05:5434'
with open(URL_FILENAME) as hostname_file:
    url = hostname_file.readline()
    hostname, port = split_URL_port(url)

with open(CREDENTIALS_FILENAME) as database_credentials_file:
    username, password = [line.strip() for line in database_credentials_file][:2]

# import sys
# sys.path.append(r"C:\Git_Repos\nbs")
# sys.path.append(r"C:\Git_Repos\bruty")
from fuse_dev.fuse.meta_review.meta_review import MetadataDatabase, database_has_table, split_URL_port
from data_management.db_connection import connect_with_retries
hostname, port = split_URL_port(url)
table_name = 'pbc_utm19n_mllw'
table_names = [('metadata', 'pbc_utm19n_mllw'), ]

connection = connect_with_retries(database='metadata', user=username, password=password, host=hostname, port=port)
cursor = connection.cursor()

# used admin credentials for this
# cursor.execute("create table serial_19n_mllw as (select * from pbc_utm19n_mllw)")
# connection.commit()
# cursor.execute('ALTER SEQUENCE serial_19n_mllw_sid_seq RESTART WITH 10000')
# cursor.execute("update serial_19n_mllw set sid=sid+10000")
# connection.commit()

cursor.execute(f'SELECT * FROM {table_name}')
records = cursor.fetchall()
fields = [desc[0] for desc in cursor.description]

fp = open("e:\\data\\nbs\\" + table_name + ".pickle", "wb")
pickle.dump(records, fp)
pickle.dump(fields, fp)
fp.close()

