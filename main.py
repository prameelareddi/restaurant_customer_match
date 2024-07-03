import cv2
import os
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from compare_faces_dev import filter_faces
import glob
import datetime
from imutils import paths
import pickle
import imutils
import dlib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from fx import prewhiten, l2_normalize
from keras.models import load_model
from scipy.spatial import distance
from compare_faces_dev import filter_facesimport
import numpy as np


detector = MTCNN()
import time

mins_count=21

model_path = './data/model/facenet_keras.h5'
font_path = './data/font/Calibri Regular.ttf'
embedding_path = './data/arrays/embeddings.npz'
vars_path = './data/arrays/vars.npz'
model = load_model(model_path,compile=False)

video_sources = cv2.VideoCapture('rtsp://admin:inndata123@10.10.5.202:554/cam/realmonitor?channel=1&subtype=0')
frame_count = 0
while 1:

    curr_time = datetime.datetime.now()
    mins = int(curr_time.strftime('%M'))
    var, frame = video_sources.read()
    # h, w, _ = frame.shape

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # frame = cv2.resize(frame, (1500, 1000))
    faces = detector.detect_faces(rgb_small_frame)  # result
    for result in faces:
        x_face, y_face, w_face, h_face = result['box']
        x_face = x_face * 4
        y_face = y_face * 4
        w_face = w_face * 4
        h_face = h_face * 4
        image = frame[y_face:y_face + h_face, x_face:x_face + w_face]
        output_name = "{}".format(f'{frame_count}.jpg', x_face, y_face, w_face, h_face)
        cv2.imwrite('imgs/' + output_name, cv2.resize(image,(350,350)))
        frame_count += 1

        # Margins for Face box
        dw = 0.1 * w_face
        dh = 0.2 * h_face
        dist = []
        #condition for retrain the model
        if mins == mins_count:
            stat_time=time.time()
            filter_faces()
            end_time=time.time()
            print(end_time-stat_time)
        if os.path.exists('./data/arrays/embeddings.npz') == True:
            loaded_embeddings = np.load(embedding_path)
            embeddings, names = loaded_embeddings['a'], loaded_embeddings['b']
            loaded_vars = np.load(vars_path)
            slope, intercept = loaded_vars['a'], loaded_vars['b']
            for i in range(len(embeddings)):
                dist.append(distance.euclidean(l2_normalize(model.predict(prewhiten(
                    cv2.resize(frame[y_face:y_face + h_face, x_face:x_face + w_face], (160, 160)).reshape(-1, 160,
                                                                                                          160, 3)))),
                    embeddings[i].reshape(1, 128)))
            dist = np.array(dist)
            if dist.min() <= .85:
                name = names[dist.argmin()]
                #get visited count also
            else:
                name = 'Unidentified'
            if name != 'Unidentified':
                font_size = int(slope[dist.argmin()] * ((w_face + 2 * dw) // 3) * 2 + intercept[dist.argmin()])
            else:
                font_size = int(0.1974311 * ((w_face + 2 * dw) // 3) * 2 + 0.03397702412218706)

            font = ImageFont.truetype(font_path, font_size)
            size = font.getbbox(name)

            cv2.rectangle(frame,
                          pt1=(x_face - int(np.floor(dw)), (y_face - int(np.floor(dh)))),
                          pt2=((x_face + w_face + int(np.ceil(dw))), (y_face + h_face + int(np.ceil(dh)))),
                          color=(0, 255, 0),
                          thickness=2)  # Face Rectangle

            cv2.rectangle(frame,
                          pt1=(x_face - int(np.floor(dw)), y_face - int(np.floor(dh)) - size[1]),
                          pt2=(x_face + size[0], y_face - int(np.floor(dh))),
                          color=(0, 255, 0),
                          thickness=-1)
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            draw.text((x_face - int(np.floor(dw)), y_face - int(np.floor(dh)) - size[1]), name, font=font,
                      fill=(255, 0, 0))
            frame = np.array(img)

    cv2.imshow('Frame', cv2.resize(frame, (800, 600)))

    if cv2.waitKey(30) & 255 == ord('q'):
        break


video_sources.release()
cv2.destroyAllWindows()






