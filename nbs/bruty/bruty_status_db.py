import os
import platform
from copy import copy
from contextlib import contextmanager

from nbs.bruty.nbs_postgres import connection_with_retries

class BrutyStatusDB:
    """
    Database access for Bruty process status and controls.

    Provides reusable connection handling and methods for
    reading process-control requests from public.bruty_status.
    """

    def __init__(self, conn_info):
        self.conn_info = copy(conn_info)

    @contextmanager
    def cursor(self):
        """
        Open a database connection and cursor.

        Commit successful operations, roll back on errors,
        and always close the cursor and connection.
        """
        connection, cursor = connection_with_retries(
            self.conn_info
        )

        try:
            yield cursor
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            cursor.close()
            connection.close()

    def get_request_exit(self, description, pid=None):
        """
        Check whether a Bruty process has an exit request.

        Parameters
        ----------
        description : str
            Process type: 'Bruty Manager', 'Combine',
            or 'Export'.
        pid : int, optional
            Process ID to check. Defaults to the current
            process ID.

        Returns
        -------
        bool
            True if an exit was requested, otherwise False.
        """
        if pid is None:
            pid = os.getpid()

        with self.cursor() as cursor:
            cursor.execute(
                """
                SELECT request_exit
                FROM public.bruty_status
                WHERE pid = %s
                  AND host = %s
                  AND description = %s
                ORDER BY status_id DESC
                LIMIT 1
                """,
                (pid, platform.node(), description)
            )

            row = cursor.fetchone()

        return bool(row[0]) if row else False