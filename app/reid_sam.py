from openvino.inference_engine import IENetwork, IEPlugin
import cv2
import numpy as np
import platform
from face_utils import align_face

class FaceReIdentification():
    def __init__(self):
        model_xml = "./extension/IR/FP32/face-reidentification-retail-0095.xml"
        model_bin = "./extension/IR/FP32/face-reidentification-retail-0095.bin"
        net = IENetwork(model=model_xml, weights=model_bin)
        if platform.system()  == 'Darwin':
            self.cpu_extension = 'extension/libcpu_extension.dylib'
        else:
            self.cpu_extension = 'extension/cpu_extension.dll'
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        self.plugin = IEPlugin(device='CPU', plugin_dirs=None)
        self.plugin.add_cpu_extension(self.cpu_extension)
        self.exec_net = self.plugin.load(network=net, num_requests=2)
        self.facial_landmark = FacialLamdmark()
        self.aligned_face = self.facial_landmark.turn_face()


    def get_feature_vec(self, face_img):
        # img = cv2.imread(face_img)
        in_frame = cv2.resize(face_img, (self.w, self.h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        self.exec_net.infer(inputs={self.input_blob: in_frame})
        res = self.exec_net.requests[0].outputs[self.out_blob] # (1, 256, 1, 1)
        feature_vec = res.reshape(1, 256)

        return feature_vec[0]

    def cos_similarity(self):
        # face_img = './kai3.jpg'
        face_img = './ryusukeSam.jpg'

        img = cv2.imread(face_img)
        X = self.get_feature_vec(img)
        Y = self.get_feature_vec(self.aligned_face)

        # (1, 256) x (256, n) = (1, n)
        return np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y, axis=0))


class FacialLamdmark():
    def __init__(self):
        model_xml = "./extension/IR/FP32/landmarks-regression-retail-0009.xml"
        model_bin = "./extension/IR/FP32/landmarks-regression-retail-0009.bin"
        net = IENetwork(model=model_xml, weights=model_bin)
        self.cpu_extension = './extension/libcpu_extension.dylib'
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        self.plugin = IEPlugin(device='CPU', plugin_dirs=None)
        self.plugin.add_cpu_extension(self.cpu_extension)
        self.exec_net = self.plugin.load(network=net, num_requests=2)
        self.face_id = 0
        self.face_img1 = './kai4.jpg'

    def turn_face(self):
        face = cv2.imread(self.face_img1)
        facial_landmarks = np.zeros((face.shape[2], 5, 2))
        in_frame = cv2.resize(face, (self.w, self.h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        self.exec_net.start_async(request_id=0 ,inputs={self.input_blob: in_frame})

        if self.exec_net.requests[0].wait(-1) == 0:
            res = self.exec_net.requests[0].outputs[self.out_blob].reshape(1, 10)[0]

            lm_face = face.copy()
            for i in range(res.size // 2):
                normed_x = res[2 * i]
                normed_y = res[2 * i + 1]
                x_lm = lm_face.shape[1] * normed_x
                y_lm = lm_face.shape[0] * normed_y
                cv2.circle(lm_face, (int(x_lm), int(y_lm)), 1 + int(0.03 * lm_face.shape[1]), (255, 255, 0), -1)
                facial_landmarks[self.face_id][i] = (x_lm, y_lm)

            aligned_face = face.copy()
            aligned_face = align_face(aligned_face, facial_landmarks[self.face_id])
            cv2.imwrite('out.jpg', aligned_face)

            return aligned_face


if __name__ == "__main__":
    face_reidfy = FaceReIdentification()
    # print(face_reidfy.get_feature_vec('./ryusukeSam.jpg'))
    print(face_reidfy.cos_similarity())
    # facel_landmark = FacialLamdmark()
    # print(facel_landmark.turn_face())