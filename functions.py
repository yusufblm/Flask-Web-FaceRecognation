from decimal import Decimal
from json import JSONEncoder
import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_min_distance(distances):
    min_distance_index = np.argmin(distances)
    index = int(min_distance_index)
    print("min_distance: ", distances[index])
    return index


def p_root(value, root):
    root_value = 1 / float(root)
    return round(Decimal(value) **
                 Decimal(root_value), 5)


def minkowski_distance(x, y, p_value):
    # pass the p_root function to calculate
    # all the value of vector parallely
    return (p_root(sum(pow(abs(a - b), p_value)
                       for a, b in zip(x, y)), p_value))


def user_or_admin_insert_logs(tcNo, username):
    try:
        from MySQL_Connector import connect_to_database
        connection, cursor = connect_to_database()
        cursor.execute("INSERT INTO USERS_LOGS(user_TC,user_name) VALUES (%s,%s)", (tcNo, username))
        connection.commit()
        connection.close()
        return True
    except Exception as e:
        print(f'Hata oluştu user logs {e}')
        return False


def user_or_admin_insert_logs_for_email(email):
    try:
        from MySQL_Connector import connect_to_database
        connection, cursor = connect_to_database()
        cursor.execute("select user_TC, firstname, lastname from USERS where email=%s", (email,))
        data = cursor.fetchall()[0]
        tcNo = data[0]
        username = data[1] + " " + data[2]
        cursor.execute("INSERT INTO USERS_LOGS(user_TC,user_name) VALUES (%s,%s)", (tcNo, username))
        connection.commit()
        connection.close()
        return True
    except Exception as e:
        print(f'Hata oluştu user logs {e}')
        return False
