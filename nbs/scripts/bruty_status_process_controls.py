from nbs.bruty.nbs_postgres import (
    connect_params_from_config,
    connection_with_retries,
)

from nbs.configs import get_logger

LOGGER = get_logger('nbs.bruty.insert')


def get_enc_request_pause(config, status_id):
    conn_info = connect_params_from_config(config)
    conn_info.database = config.get('export_database')

    connection, cursor = connection_with_retries(conn_info)

    try:
        cursor.execute(
            """
            SELECT enc_request_pause
            FROM public.bruty_status
            WHERE status_id = %s
            """,
            (status_id,)
        )

        row = cursor.fetchone()

        return False if row is None else row[0]

    finally:
        cursor.close()
        connection.close()


def update_is_paused(config, status_id, is_paused):
    conn_info = connect_params_from_config(config)
    conn_info.database = config.get('export_database')

    connection, cursor = connection_with_retries(conn_info)

    try:
        if is_paused:
            cursor.execute(
                """
                UPDATE public.bruty_status
                SET is_paused = TRUE,
                    is_paused_at = clock_timestamp()
                WHERE status_id = %s
                """,
                (status_id,)
            )
        else:
            cursor.execute(
                """
                UPDATE public.bruty_status
                SET is_paused = FALSE
                WHERE status_id = %s
                """,
                (status_id,)
            )

        connection.commit()

    finally:
        cursor.close()
        connection.close()


def update_global_is_paused(config):
    conn_info = connect_params_from_config(config)
    conn_info.database = config.get('export_database')

    connection, cursor = connection_with_retries(conn_info)

    try:
        # Check the global pause request.
        cursor.execute(
            """
            SELECT enc_request_pause, is_paused
            FROM public.bruty_status
            WHERE status_id = 1
              AND description = 'Global Bruty'
            FOR UPDATE
            """
        )

        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Global Bruty status row not found.")

        global_pause_requested, currently_paused = row

        # Check whether all registered managers are paused.
        cursor.execute(
            """
            SELECT COUNT(*), BOOL_AND(is_paused)
            FROM public.bruty_status
            WHERE description = 'Bruty Manager'
            """
        )

        manager_count, all_managers_paused = cursor.fetchone()

        global_is_paused = (
            global_pause_requested
            and manager_count > 0
            and all_managers_paused is True
        )

        # Update only when the global state changes.
        if global_is_paused != currently_paused:
            cursor.execute(
                """
                UPDATE public.bruty_status
                SET is_paused = %s,
                    is_paused_at = CASE
                        WHEN %s THEN clock_timestamp()
                        ELSE is_paused_at
                    END
                WHERE status_id = 1
                  AND description = 'Global Bruty'
                """,
                (global_is_paused, global_is_paused)
            )

        connection.commit()
        return global_is_paused

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def update_progress(config, status_id, progress):
    progress = max(0, min(100, int(progress)))

    conn_info = connect_params_from_config(config)
    conn_info.database = config.get('export_database')

    connection, cursor = connection_with_retries(conn_info)

    cursor.execute(
        """
        UPDATE public.bruty_status
        SET progress = %s
        WHERE status_id = %s
        """,
        (progress, status_id)
    )

    connection.commit()
    cursor.close()
    connection.close()