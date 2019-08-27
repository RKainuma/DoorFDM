from openvino.inference_engine import IENetwork, IEPlugin
import cv2
import numpy as np
import platform
from face_utils import align_face
import sys
import os
from face_detect import face_detction

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

    def get_feature_vec(self, face_img, outputfile):
        # img = cv2.imread(face_img)
        in_frame = cv2.resize(face_img, (self.w, self.h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        self.exec_net.infer(inputs={self.input_blob: in_frame})
        res = self.exec_net.requests[0].outputs[self.out_blob]
        feature_vec = res.reshape(1, 256)
        feature_vec[0].dump('face_pts/'+outputfile)

if __name__ == "__main__":
    args = sys.argv
    if len(args) >= 3:
        find_file = os.path.isfile(args[1])
        if find_file:
            ret = face_detction(args[1])
            if len(ret) != 1:
                print('Faces are None or more than 2')
            else:
                face_reidfy = FaceReIdentification()
                face_reidfy.get_feature_vec(ret[0], args[2])
                print('Done')
        else:
            print('{} does not exists'.format(args[1]))
    else:
        print('Run with inputfile and outputfile\nEx: python calc_cos_pts.py face_img.jpg tanaka.pts')
