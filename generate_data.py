import os
import numpy as np
from PIL import Image,ImageFont
from fx import prewhiten,l2_normalize
from keras.models import load_model
from sklearn.linear_model import LinearRegression
import tensorflow as tf

#faces_path = './data/faces/'
model_path = './data/model/facenet_keras.h5'
images_path='./customers/'

def generate_embeddings():
    print('generating embeddings')
    names = []
    dirs=[]

    for it in os.scandir('./customers'):
        if it.is_dir():
            dirs.append(it.path)
    for dir in dirs:
        for i, fn in enumerate(os.listdir(dir)):
            names.append(fn)
    if len(names)==0:
        print("No Face Found")
        input()
        quit()
    names.sort()
    names=np.array(names)
    faces=[]
    for dir in dirs:
        for i in names:
            if os.path.exists(dir+'/'+i)==True:
            #160
                img=Image.open(dir+'/'+i).resize((160,160))
                img=np.array(img)
                faces.append(img)

    faces=np.array(faces)
    faces=prewhiten(faces)
    model=load_model(model_path)
    embs=model.predict(faces)
    embs=l2_normalize(embs)

    for i in range(len(names)):
        names[i] = names[i].split('_')[0]

    font_path='./data/font/Calibri Regular.ttf'

    slope=[]
    intercept=[]
    for i in names:
        x=[]
        y=[]
        for j in range(1,100):
            font=ImageFont.truetype(font_path,j)
            x.append(j)
            y.append(font.getbbox(i)[0])
        lin=LinearRegression().fit(np.array(y).reshape(-1,1),np.array(x))
        slope.append(lin.coef_)
        intercept.append(lin.intercept_)
    slope=np.array(slope)
    intercept=np.array(intercept)
    embeddings=os.listdir('./data/arrays/')
    for file in embeddings:
        os.remove('./data/arrays/'+file)
    np.savez_compressed('./data/arrays/vars.npz',a=slope,b=intercept)
    np.savez_compressed('./data/arrays/embeddings.npz',a=embs,b=names)
