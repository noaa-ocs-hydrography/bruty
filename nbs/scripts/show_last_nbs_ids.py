from nbs.bruty.nbs_postgres import get_debug_connection, get_tablenames, show_last_ids

if __name__ == "__main__":
    conn_info, connection, cursor, config = get_debug_connection()
    tablenames = get_tablenames(cursor)
    show_last_ids(cursor, tablenames)
