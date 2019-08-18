from flask import Flask, Response, render_template, request
from camera import VideoCamera
import os
import sys
import platform
import interactive_detection

app = Flask(__name__)

detections = interactive_detection.Detections()


def gen(camera):
    while True:
        frame = camera.get_frame()
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
    app.run(host='0.0.0.0', threaded=True)
