import numpy as np
import MNN
import cv2
import time
from head_pose import get_head_pose
from tracker import Tracker


class FaceEmbedder:
    def __init__(self, detection_size=(160, 160), mean_RGB=(128, 128, 128)):
        self.model = cv2.dnn.readNetFromONNX("./models/casiawebface.onnx")
        self.detection_size = detection_size
        self.mean_RGB = mean_RGB

    def embed(self, frame):
        faceBlob = cv2.dnn.blobFromImage(frame, 1.0 / 128, self.detection_size, self.mean_RGB, swapRB=True, crop=False)
        self.model.setInput(faceBlob)
        face_encoding = self.model.forward()
        return face_encoding


import os
import pathlib

class Detector:
    def __init__(self, detection_size=(160, 160)):
        self.interpreter = MNN.Interpreter('./models/slim_160_latest.mnn')
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)
        self.detection_size = detection_size
        self.tracker = Tracker()

    def crop_image(self, orig, bbox):
        bbox = bbox.copy()
        image = orig.copy()
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        face_width = (1 + 2 * 0.2) * bbox_width
        face_height = (1 + 2 * 0.2) * bbox_height
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        bbox[0] = max(0, center[0] - face_width // 2)
        bbox[1] = max(0, center[1] - face_height // 2)
        bbox[2] = min(image.shape[1], center[0] + face_width // 2)
        bbox[3] = min(image.shape[0], center[1] + face_height // 2)
        bbox = bbox.astype(np.int)
        crop_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        h, w, _ = crop_image.shape
        crop_image = cv2.resize(crop_image, self.detection_size)
        return crop_image, ([h, w, bbox[1], bbox[0]])

    def detect(self, img, bbox):
        crop_image, detail = self.crop_image(img, bbox)
        crop_image = (crop_image - 127.0) / 127.0
        crop_image = np.array([np.transpose(crop_image, (2, 0, 1))]).astype(np.float32)
        tmp_input = MNN.Tensor((1, 3, *self.detection_size), MNN.Halide_Type_Float, crop_image,
                               MNN.Tensor_DimensionType_Caffe)
        self.input_tensor.copyFrom(tmp_input)
        start = time.time()
        self.interpreter.runSession(self.session)
        raw = np.array(self.interpreter.getSessionOutput(self.session).getData())
        end = time.time()
        # print("MNN Inference Time: {:.6f}".format(end - start))
        try:
            landmark = raw[0:136].reshape((-1, 2))
        except:
            landmark = raw[0, 0:136].reshape((-1, 2))
            pass
        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3]
        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2]
        landmark = self.tracker.track(img, landmark)
        _, PRY_3d = get_head_pose(landmark, img)
        return landmark, PRY_3d[:, 0]
