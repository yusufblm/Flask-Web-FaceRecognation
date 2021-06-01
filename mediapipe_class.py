import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_draw


class MediaPipePose:

    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.70, min_tracking_confidence=0.5, upper_body_only=True)
        self.MAX_OCCS = 45
        self.number_of_occs = np.zeros(self.MAX_OCCS, dtype=np.int8)
        self.ci = 0

    def close(self):
        self.pose.close()

    def get_pose(self, image):
        image_width, image_height, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        is_frame_occs = 0
        y = []
        if results.pose_landmarks is not None:
            for lm in results.pose_landmarks.landmark[13:23]:
                y.append(min(math.floor(lm.y * image_width), image_width - 1))

            y = int(np.min(y))

            # cv2.circle(image,(50,y),6,(0,0,255),-1)
            cv2.line(image, (50, y), (50, y + 20), (0, 0, 255), 5);

            if y < (image_width - image_width // 50):
                # cv2.putText(image,'FAKE',(50,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                is_frame_occs = 1

        self.number_of_occs[self.ci % self.MAX_OCCS] = is_frame_occs
        self.ci += 1

        if np.mean(self.number_of_occs) > 0.10:
            CHECK_ALIVE_TEST = False
            # cv2.circle(image,(image_height//10,image_width//10),12,(0,0,255),-1)
        else:
            CHECK_ALIVE_TEST = True
            # cv2.circle(image,(image_height//10,image_width//10),12,(0,255,0),-1)

        return image, CHECK_ALIVE_TEST


class MediaPipeFaceMesh:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp_draw
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.7,
                                                    min_tracking_confidence=0.5, max_num_faces=4)
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=0)
        self.NumFaces = 0
        self.bboxes = []
        self.xAll = []
        self.yAll = []
        self.zAll = []
        self.left_eye_p = np.array([0, 0])
        self.right_eye_p = np.array([0, 0])
        self.roll_degree = 0
        self.left_eye = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398, 362]
        self.face_oval = frozenset(
            [(10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323),
             (323, 361),
             (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152), (152, 148),
             (148, 176), (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234),
             (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)])

    def close(self):
        self.face_mesh.close()

    def get_face(self):

        face = self.image.copy()
        try:
            xstart = self.bboxes[0][0]
            ystart = self.bboxes[0][1]
            xstop = self.bboxes[0][2]
            ystop = self.bboxes[0][3]

            frame_face = self.image[ystart:ystop, xstart:xstop]
            w, h = frame_face.shape[0], frame_face.shape[1]

            if w < 32 or h < 32:
                is_face = 0
                return self.image, is_face

            try:
                face = np.column_stack([np.zeros((w, np.max([np.abs((w - h) // 2), 1]), 3), dtype=np.uint8), frame_face,
                                        np.zeros((w, np.max([np.abs((w - h) // 2), 1]), 3), dtype=np.uint8)])
            except:
                face = frame_face.copy()


        except Exception as e:
            print(f'mp-get_face error: {e}')

        try:
            if face.shape[0] < 32 or face.shape[1] < 32:
                is_face = 0
                return self.image, is_face
            else:
                is_face = len(self.results.multi_face_landmarks)
                return face, is_face
        except Exception as e:
            is_face = 0
            return self.image, is_face

    def get_angle_btw_eye_points(self, p1_number, p2_number):
        self.left_eye_p = np.array([self.results.multi_face_landmarks[0].landmark[p1_number].x,
                                    self.results.multi_face_landmarks[0].landmark[p1_number].y])
        self.right_eye_p = np.array([self.results.multi_face_landmarks[0].landmark[p2_number].x,
                                     self.results.multi_face_landmarks[0].landmark[p2_number].y])
        deg_rad = math.atan2(self.left_eye_p[1] - self.right_eye_p[1], self.left_eye_p[0] - self.right_eye_p[0])
        return (180 - math.degrees(deg_rad))

    def get_angle_btw_forehead_nose_points(self, p1_number, p2_number):
        self.left_eye_p = np.array([self.results.multi_face_landmarks[0].landmark[p1_number].y,
                                    self.results.multi_face_landmarks[0].landmark[p1_number].z])
        self.right_eye_p = np.array([self.results.multi_face_landmarks[0].landmark[p2_number].y,
                                     self.results.multi_face_landmarks[0].landmark[p2_number].z])
        deg_rad = math.atan2(self.left_eye_p[1] - self.right_eye_p[1], self.left_eye_p[0] - self.right_eye_p[0])
        tmp = math.degrees(deg_rad)
        if tmp < 0:
            return 180 + tmp
        else:
            return 180 - tmp

    def get_face_coordinates_box(self):
        image_width, image_height, _ = self.image.shape
        self.bboxes = []
        try:
            for idx_face, face_landmarks in enumerate(self.results.multi_face_landmarks):

                self.roll_degree = self.get_angle_btw_eye_points(159, 386)
                self.pitch_degree = self.get_angle_btw_forehead_nose_points(9, 0)

                x_chk_left = self.results.multi_face_landmarks[0].landmark[234].x
                x_chk_right = self.results.multi_face_landmarks[0].landmark[454].x
                tmp = np.abs(x_chk_right - x_chk_left) * image_height
                w_roi = int(np.max([tmp // 10, 2]))
                x = [];
                y = []
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x.append(min(math.floor(landmark.x * image_height), image_height - 1))
                    y.append(min(math.floor(landmark.y * image_width), image_width - 1))

                x_start = np.max([np.min(x), 1])
                y_start = np.max([np.min(y) - w_roi, 1])

                x_stop = np.min([np.max(x), image_height])
                y_stop = np.min([np.max(y) + w_roi // 2, image_width])

                self.bboxes.append([x_start, y_start, x_stop, y_stop])
        except Exception as e:
            # print(f'get face boxes - MP - error: {e} ')
            pass

        self.bboxes = np.array(self.bboxes)
        return self.bboxes

    def get_forehead(self):
        hh = self.image.shape[0];
        ww = self.image.shape[1]

        x = self.results.multi_face_landmarks[0].landmark[10].x
        y = self.results.multi_face_landmarks[0].landmark[10].y

        x = int(x * ww);
        y = int(y * hh)

        x_chk_left = self.results.multi_face_landmarks[0].landmark[234].x
        y_chk_left = self.results.multi_face_landmarks[0].landmark[234].y

        x_chk_right = self.results.multi_face_landmarks[0].landmark[454].x
        y_chk_right = self.results.multi_face_landmarks[0].landmark[454].y

        tmp = np.abs(x_chk_right - x_chk_left) * ww
        w_roi = int(np.max([tmp // 12, 2]))

        pt1 = (x - 3 * w_roi, y)
        pt2 = (x + 3 * w_roi, y + 2 * w_roi)

        rc = self.image[pt1[1]:pt2[1], pt1[0]:pt2[0], 0];
        rc_mean = np.mean(rc)
        gc = self.image[pt1[1]:pt2[1], pt1[0]:pt2[0], 1];
        gc_mean = np.mean(gc)
        bc = self.image[pt1[1]:pt2[1], pt1[0]:pt2[0], 2];
        bc_mean = np.mean(bc)
        return pt1, pt2, rc_mean, gc_mean, bc_mean

    def get_eye(self):
        x = [];
        y = [];
        z = [];
        x = np.zeros(17);
        y = np.zeros(17);
        eye = self.image.copy()
        try:
            for lm in self.results.multi_face_landmarks:
                for ii, val in enumerate(self.left_eye):
                    x[ii] = self.results.multi_face_landmarks[0].landmark[val].x
                    y[ii] = self.results.multi_face_landmarks[0].landmark[val].y

            hh = self.image.shape[0];
            ww = self.image.shape[1]
            x = x * ww
            y = y * hh

            xstart = int(np.min(x))
            ystart = int(np.min(y))
            xstop = int(np.max(x))
            ystop = int(np.max(y))
            w = np.abs(xstop - xstart)
            h = np.abs(ystop - ystart)
            x = xstart + w // 2
            y = ystart + h // 2
            w = np.max([w, h, 2]) // 2
            # image = cv2.rectangle(image, (x-w//2,y-w//2), (x+w//2,y+w//2), (255,0,0), 3)
            eye = self.image[np.max([y - w // 2, 0]):y + w // 2, np.max([x - w // 2, 0]):x + w // 2]
            eye = cv2.resize(eye, (150, 150))
        except Exception as e:
            print(f'eye shape: {eye.shape}')
            print(e)
        return eye

    def detect_face(self, image):
        tmp = image.copy()
        self.image = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        self.image.flags.writeable = False
        self.results = self.face_mesh.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

        try:
            n = len(self.results.multi_face_landmarks)
        except:
            n = 0
        self.NumFaces = n
        return n

    def draw_face(self):
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=self.image,
                    landmark_list=face_landmarks,

                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)
        return self.image
