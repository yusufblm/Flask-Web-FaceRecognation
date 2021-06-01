import json
import threading
import time
import warnings
from io import BytesIO
from wsgiref.util import FileWrapper

import cv2
import imutils
import jsonpickle
import numpy as np
from flask import Response

import configs
from MySQL_Connector import MySQL
from detector import Detector
from detector import FaceEmbedder
from face_detector import Detector as FaceDetector
from functions import NumpyArrayEncoder, minkowski_distance, get_min_distance, user_or_admin_insert_logs_for_email, \
    user_or_admin_insert_logs

stop_threads = False
maxAngle = 0
start_verify_time = 0
finish_verify_time = 0
insan_sayisi = 0
frame = None
outputFrame = None
bboxes = None
lock = threading.Lock()
lmk_detector = Detector()
face_detector = FaceDetector()
warnings.simplefilter("ignore", DeprecationWarning)
face_embedder = FaceEmbedder()
camera_obj: cv2.VideoCapture() = cv2.VideoCapture()
mysql = MySQL()


# from MySQL_Connector import connect_to_database


def cam_open():
    global camera_obj
    camera_obj = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # adres = "rtsp://yusuf:123456@192.168.1.50:8080/h264_ulaw.sdp"
    # camera_obj = cv2.VideoCapture(adres, cv2.CAP_FFMPEG)

    if camera_obj.isOpened():
        return True
    else:
        return False


def cam_close():
    global camera_obj
    if camera_obj.isOpened():
        camera_obj.release()
        return True
    else:
        return False


def detect_faces_from_cam():
    global stop_threads, frame, bboxes, insan_sayisi, outputFrame, lock, finish_verify_time, maxAngle, start_verify_time
    while True:
        try:
            if camera_obj is not None and camera_obj.isOpened():
                ret, frame = camera_obj.read()
                frame = imutils.resize(frame, width=configs.CAM_RES_WIDTH)
                bboxes, _ = face_detector.detect(frame)
                insan_sayisi = bboxes.shape[0]
                with lock:
                    outputFrame = frame.copy()
                    try:
                        for i in range(0, insan_sayisi):
                            cv2.rectangle(outputFrame, (bboxes[i][0], bboxes[i][1]), (bboxes[i][2], bboxes[i][3]),
                                          (0, 0, 255), 2)
                    except Exception as e:
                        print(f"Hata {e}")
                        pass
                if stop_threads:
                    break
            else:
                insan_sayisi = 0
                time.sleep(configs.CLOSE_CAM_SLEEP_TIME)
                pass
        except:
            # print(f"detect_faces_from_cam - error: {e}")
            pass


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            encodedImage = encodedImage.tobytes()
            if not flag:
                continue
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'


def get_face_embed_vector():
    global bboxes, frame
    frame_copy = frame.copy()
    start_verify_time = time.time()
    maxAngleFace = 180
    time_dif = 0
    while time_dif < configs.FACE_EMBED_MAX_TIME:
        time_dif = time.time() - start_verify_time
        print(f'time_dif: {time_dif}')
        try:
            if bboxes.shape[0] > 0:
                bbox, cx, cy, ww, hh, w = get_box_coordinates(bboxes)
                try:
                    lmks, PRY_3d = lmk_detector.detect(frame, bbox)
                    mean_face_angle = np.mean(np.abs(PRY_3d))
                    if mean_face_angle < maxAngleFace:
                        maxAngleFace = mean_face_angle
                        frame_copy = frame.copy()
                    # print(f'angles: {PRY_3d}, mean: {mean_face_angle}')
                except Exception as e:
                    print("Exception error embed: ", e)

                    # frame_face = frame[np.max((cx-w,0)):cx+w, np.max((cy-w,0)):cy+w]
                frame_face = frame_copy[np.max(
                    (cx - ww // 2, 0)):cx + ww // 2, np.max((cy - hh // 2, 0)):cy + hh // 2]
                w, h = frame_face.shape[0], frame_face.shape[1]
                frame_face = np.column_stack([np.zeros((w, np.max(((w - h) // 2, 0)), 3), dtype=np.uint8), frame_face,
                                              np.zeros((w, np.max(((w - h) // 2, 0)), 3), dtype=np.uint8)])

                if frame_face.shape[0] > 0 and maxAngleFace <= configs.FACE_MEAN_MAX_ANGLE:
                    vector_person = face_embedder.embed(frame_face)
                    vector_person_encoded = json.dumps(
                        vector_person, cls=NumpyArrayEncoder)
                    return vector_person_encoded, frame_face, maxAngleFace
        except Exception:
            pass
    return None, None, maxAngleFace


# noinspection PyShadowingNames
def get_box_coordinates(bboxes):
    bbox = bboxes[0]
    bbox = bbox.astype(np.int)
    # Only Face
    cx = (bbox[1] + bbox[3]) // 2
    cy = (bbox[0] + bbox[2]) // 2
    ww = (bbox[3] - bbox[1])
    hh = (bbox[2] - bbox[0])
    w = np.max((ww, hh)) // 2
    return bbox, cx, cy, ww, hh, w


def register(data=None):
    global frame, mysql
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
    except Exception:
        print("frame not ready")
        time.sleep(2)
        _, img_encoded = cv2.imencode('.jpg', frame)

    vector_person_encoded, frame_face, maxAngleFace = get_face_embed_vector()

    if maxAngleFace > configs.FACE_MEAN_MAX_ANGLE and maxAngleFace != 180:
        return return_response(message='FAILED FACE ANGLE ERROR', status_code=200, code=4)

    if vector_person_encoded is not None:

        user_TC = data['tc']
        user_name = data['firstname']
        user_lastname = data['lastname']
        # birthdate = data['birthdate']
        user_role = data['user_role']
        email = data['email']
        password = data['password']
        image_data = img_encoded.tostring()
        user = (user_TC, user_name, user_lastname,
                user_role, email, password, vector_person_encoded, image_data)
        insert_query = "INSERT INTO USERS (user_TC, firstname, lastname, user_role, email, password, vector, image_data) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"

        # try:
        #     connection, cursor = connect_to_database()
        # except:
        #     return return_response(message='FAILED, DATABASE CONNECTION ERROR', status_code=200, code=2)
        #     pass
        if mysql.check_connection():
            pass
        else:
            return return_response(message='FAILED, DATABASE CONNECTION ERROR', status_code=200, code=2)

        mysql.execute(
            '''SELECT firstname,lastname  FROM USERS WHERE user_TC = %s''', (user_TC,))
        ci = 0
        for user_name, user_lastname in mysql.fetchall():
            print(user_name + user_lastname)
            ci += 1

        if ci == 0:
            try:
                mysql.execute(insert_query, user)
                mysql.commit()
                return return_response(message="SUCCESS", status_code=200, code=0)

            except Exception as e:
                print("Exception error: ", e)
                return return_response(message='FAILED NO FACE', status_code=200, code=1)
        else:
            user_name = data['firstname']
            user_lastname = data['lastname']
            mysql.execute(
                '''UPDATE USERS SET firsname=%s, lastname=%s,password=%s, vector=%s, image_data=%s WHERE user_TC = %s''',
                (user_name, user_lastname, password, vector_person_encoded, image_data, user_TC))
            mysql.commit()
            mysql.close()
            return return_response(message='SUCCESS RECORD UPDATED', status_code=200, code=3)

    else:
        return return_response(message='FAILED NO FACE', status_code=200, code=1)


def return_response(message: str, status_code: int, code: int, result=None, user_log=None) -> Response:
    if result is None:
        response = {'message': message, 'code': code}
    else:
        response = {'message': message, 'code': code, 'result': result, 'user_log':user_log}
    return Response(response=jsonpickle.encode(response), status=status_code, mimetype="application/json")


def person_recognition():  # dışarda çağırılacak aynı
    global mysql
    vector_person_encoded, frame_face, maxAngleFace = get_face_embed_vector()
    start = time.time()
    if maxAngleFace > configs.FACE_MEAN_MAX_ANGLE and maxAngleFace != 180:
        response = {'message': 'FAILED FACE ANGLE ERROR', 'code': 5}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

    if vector_person_encoded is not None:

        try:
            mysql.check_connection()
        except:
            return return_response(message='FAILED, DATABASE CONNECTION ERROR', status_code=200, code=2)

        distances = []

        mysql.execute("SELECT user_TC, firstname, vector FROM USERS")

        # cursor.execute('''UPDATE users2 SET vector = %s  WHERE U_UID = %s''', (vector_person_encoded, UUID))
        ci = 0
        for TC, person_name, row in mysql.fetchall():
            ci += 1
            arr1 = np.asarray(json.loads(row), dtype=np.float32)
            arr2 = np.asarray(json.loads(
                vector_person_encoded), dtype=np.float32)
            a1 = arr1[0]
            b1 = arr2[0]
            match_distance_1 = minkowski_distance(
                a1, b1, configs.MINKOWSKI_DISTANCE_DEG)
            distances.append(match_distance_1)
            # print("person_name: ", person_name, " : ",match_distance_1)
        if ci > 0:
            index_1 = get_min_distance(distances)
            if distances[index_1] <= configs.CONF_THRESHOLD_1_TO_N:
                # and distances2[index_2]<0.75 and index_1 == index_2:
                end = time.time()
                mysql.execute(
                    '''SELECT user_TC, firstname,lastname FROM USERS LIMIT %s, 1''', (index_1,))
                tmp = mysql.fetchall()
                print("**************")
                print(tmp)
                print("**************")
                user_TC = str(tmp[0][0])
                user_name = tmp[0][1]
                user_lastname = tmp[0][2]
                # print(user_lastname)
                # sonuc = str(abs(start - end))
                user = user_name + " " + user_lastname + "_" + user_TC
                print(f'Bulunan Kişi {user_name}')
                username = user_name + " " + user_lastname
                resp = user_or_admin_insert_logs(user_TC, username)
                if not resp:
                    print("Log kaydı eklenemedi")

                return return_response(message='SUCCESS', status_code=200, code=0, result=user)
            else:
                return return_response(message='FAILED, NO MATCH', status_code=200, code=1)
        else:

            return return_response(message='FAILED, NO RECORD', status_code=200, code=3)
    else:
        response = {'message': 'FAILED, NO FACE', 'code': 4}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")


def get_persons():
    global mysql
    data = None
    try:
        mysql.execute("SELECT user_TC, firstname, lastname,user_role,email FROM USERS")
        # data = mysql.fetchall()
        # print("-----------")
        # print(type(data))
        # print("-----------")
        return mysql.fetchall()
    except Exception as e:
        print(f'DB kaynaklı bir sorun oluştu {e}')
        return False


def get_image(user_TC):
    global mysql
    mysql.execute("Select image_data from USERS where user_TC=%s", (user_TC,))
    obj = mysql.fetchone()
    obj = obj[0]
    b = BytesIO(obj)
    w = FileWrapper(b)
    return Response(w, mimetype="multipart/x-mixed-replace;", direct_passthrough=True)


def get_persons_count():
    global mysql
    mysql.execute("Select COUNT(*) from USERS")
    data = mysql.fetchone()[0]
    return data
