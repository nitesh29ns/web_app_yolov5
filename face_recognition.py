import cv2
import numpy as np
import torch


CUSTOM_MODEL="my_face_recognition_model"
PRE_TRAIN_MODEL="yolov5s"



def generate_frame_detection(model_name:str):

    torch.device('cpu')
    if model_name == CUSTOM_MODEL:
        model = torch.hub.load('./yolov5/', 'custom', path='my_face_detection_model.pt', source='local') 
    elif model_name == PRE_TRAIN_MODEL:
        model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)

    camera=cv2.VideoCapture(0)
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            result = model(frame)

            ret,buffer=cv2.imencode('.jpg',  np.squeeze(result.render()))
            frame=buffer.tobytes()

            yield(b'--frame\r\n'
                    b'content-Type: image/jpeg\r\n\r\n'+ frame + b'\r\n')
    camera.release()
    cv2.destroyAllWindows()


