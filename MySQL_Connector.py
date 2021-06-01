import mysql.connector as sql
import mysql.connector

host = 'localhost'
user = 'root'
password = 'root'
config = {
    'host': "localhost",
    'user': "root",
    'password': "root",
    'database': "dockerMysql",
    'port': '3306'
}


def connect_to_database():
    config = {
        'host': "localhost",
        'user': "root",
        'password': "root",
        'database': "dockerMysql",
        'port': '3306'
    }
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    return connection, cursor


class MySQL:
    def __init__(self):
        self.__mydb = sql.connect(**config)
        self.__cur = self.__mydb.cursor()

    def execute(self, sql_query, val=None):
        if val is None:
            self.__cur.execute(sql_query)
        else:
            self.__cur.execute(sql_query, val)
        return self.__cur

    def executemany(self, sql_query, val_list):
        self.__cur.executemany(sql_query, val_list)

    def close(self):
        self.__mydb.close()

    def commit(self):
        self.__mydb.commit()

    def fetchall(self):
        return self.__cur.fetchall()

    def fetchone(self):
        return self.__cur.fetchone()

    def fetchmany(self):
        return self.__cur.fetchmany()

    def check_connection(self):
        try:
            self.__cur.execute("SHOW TABLES")
            print(self.fetchall())
            return True
        except Exception as e:
            print(f"Database de bir hata var {e}")
            return False
