CONF_THRESHOLD_1_TO_1 = 0.80
CONF_THRESHOLD_1_TO_N = 0.80
ALIVE_TEST_FACE_ANGLE = 45
ALIVE_TEST_TIME = 5
FACE_EMBED_MAX_TIME = 3
CLOSE_CAM_SLEEP_TIME = 1
FACE_MEAN_MAX_ANGLE = 10
MAX_NUM_OF_FACES = 10
MINKOWSKI_DISTANCE_DEG = 5
CAM_RES_WIDTH = 1024
FACE_SIZE = 0.3

LOCAL_PORT = 8000

config = {
    'host': "localhost",
    'user': "root",
    'password': "root",
    'database': "face_project",
    'port': '3306'
}

from mediapipe.python.solutions import face_mesh
