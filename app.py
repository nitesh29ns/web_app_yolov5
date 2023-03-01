from flask import Flask, request, render_template, Response
import cv2
import numpy as np
import torch
from face_recognition import generate_frame_detection

app=Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method =="POST":
        model_name = str(request.form['model'])
    return Response(generate_frame_detection(model_name=model_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template('about.html')




if __name__=="__main__":
    app.run(debug=True)

