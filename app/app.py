from flask import Flask, Response, render_template, request
from camera import VideoCamera
from logging import getLogger, basicConfig, DEBUG, INFO
import os
import sys
import platform
import interactive_detection

app = Flask(__name__)
logger = getLogger(__name__)

basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

is_async_mode = True
is_face_detection = True
is_head_pose_detection = False


def gen(camera):
    while True:
        frame = camera.get_frame(is_async_mode, is_face_detection,is_head_pose_detection)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    if platform.system() == 'Linux':
        no_v4l = False
    else:
        no_v4l = True
    camera = VideoCamera(detections, no_v4l)
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':

    devices = ['CPU',  'CPU']
    models = [None, None]

    init_prob_threshold = 0.3
    init_prob_threshold_face = 0.5
    init_plugin_dir = None
    detections = interactive_detection.Detections(
        devices, models, init_plugin_dir,
        init_prob_threshold, init_prob_threshold_face, is_async_mode)
    models = detections.models  # Get models to display WebUI.

    app.run(host='0.0.0.0', threaded=True)
