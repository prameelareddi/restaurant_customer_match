import glob
import os
import pandas as pd
import shutil
from deepface import DeepFace
from generate_data import generate_embeddings
from joblib import Parallel, delayed
import multiprocessing

import hashlib
random_data = os.urandom(128)

import mysql.connector
from mysql.connector import (connection)

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

def upload_data(customer_id,visiting_count):
    try:
        cnx = mysql.connector.connect(user='root', password='123!@#',
                                      host='10.10.5.76',
                                      database='inndata_fr')

        print(cnx.is_connected())
        if cnx.is_connected():
            cursor = cnx.cursor()
            query = "INSERT INTO customers (customer_id,visiting_count) VALUES (%s,%s)"
            val = (customer_id,visiting_count)
            cursor.execute(query, val)
            cnx.commit()
            cursor.close()
            cnx.close()
            return True
        else:
            return False

    except:
        return False
def update_data(visiting_count,folder_name):

    cnx = connection.MySQLConnection(user='root', password='123!@#',
                                     host='10.10.5.76',
                                     database='inndata_fr')
    cursor = cnx.cursor()
    query = ("UPDATE customers SET visiting_count=%s WHERE customer_id=%s")
    cursor.execute(query,(visiting_count,folder_name))
    cnx.commit()
    cursor.close()
    cnx.close()



def customer_id():

    cnx = connection.MySQLConnection(user='root', password='123!@#',
                                     host='10.10.5.76',
                                     database='inndata_fr')
    cursor = cnx.cursor()
    query = ("SELECT customer_id from customers")
    cursor.execute(query)
    result = cursor.fetchall()
    return result
    cursor.close()

def visiting_count(folder_name):

    cnx = connection.MySQLConnection(user='root', password='123!@#',
                                     host='10.10.5.76',
                                     database='inndata_fr')
    cursor = cnx.cursor()
    query = ("SELECT visiting_count from customers where customer_id = %s")
    cursor.execute(query,(folder_name,))
    result = cursor.fetchall()
    return result
    cursor.close()

def filter_imgs(dir,img):
    first_img = os.listdir('./customers/'+dir)[0]
    print('first img: ', first_img)
    result = DeepFace.verify(img1_path=img, img2_path=os.path.join('./customers/'+dir, first_img),
                             enforce_detection=False, model_name=models[-2])

    return result['verified']
def move_imgs(item,destination,folder_name,count):
    if len(item.split('/')) == 2 and len(os.listdir(destination)) < 6:
        print('similer faces: ', item)
        file_name = f'{folder_name}_{count}.jpg'
        os.rename(item, os.path.join(destination, file_name))

    else:
        print('deleting item',item)
        os.remove(item)


def filter_faces():
    df = pd.read_csv('customer_count.csv')
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    print('filtering faces')
    pkl_paths = glob.glob('imgs/*.pkl')
    for file in pkl_paths:
        os.remove(file)
    img_paths = glob.glob('imgs/*.jpg')

    while len(img_paths) != 0:
        loop_limit = 1
        count = 0
        dirs = []
        customer_data=[]
        nof_visitings=1
        img_paths = glob.glob('imgs/*.jpg')
        if len(img_paths) != 0:
            img = img_paths[0]
        else:
            break
        print('input img: ', img)
        for it in os.scandir('./customers'):
            if it.is_dir():
                dirs.append(it.path.split('/')[-1])
        print('dirs list: ', dirs)
        if len(dirs)==0:
            folder_name = '#'+hashlib.md5(random_data).hexdigest()[:16]
        elif len(dirs) != 0:
            results = Parallel(n_jobs=2)(delayed(filter_imgs)(dir,img) for dir in dirs)
            print(results)
            if True in results:
                folder_name=dirs[results.index(True)]
                print('choose existing dir: ',folder_name)

            else:
                exists = 1
                while exists:
                    folder_name = '#' + hashlib.md5(random_data).hexdigest()[:16]
                    #print(folder_name)
                    if folder_name in dirs:
                        folder_name = '#' + hashlib.md5(random_data).hexdigest()[:16]
                    if folder_name not in dirs:
                        exists = 0
                print('created: ', folder_name)

        result = DeepFace.find(img_path=img, db_path='imgs', model_name=models[1], enforce_detection=False)
        target_len = len(result['identity'][1:])
        if len(result['identity'])>6 :
            if len(visiting_count(folder_name))!=0:
                visit_count = visiting_count(folder_name)[0][0]
                visit_count += 1
                update_data(folder_name, nof_visitings)

            else:
                upload_data(folder_name, nof_visitings)
            if os.path.exists(os.path.join('customers', folder_name)) == False:
                os.makedirs(os.path.join('customers', folder_name))
            destination = os.path.join('customers', folder_name)
            file_name = f'{folder_name}_{count}.jpg'
            # shutil.move(img, os.path.join(destination, img.split('/')[1]))
            os.rename(img, os.path.join(destination, file_name))


            count = [*range(1, target_len, 1)]

            Parallel(n_jobs=3)(delayed(move_imgs)(item, destination, folder_name, count) for item, count in
                               zip(result['identity'][1:], count))

            pkl_paths = glob.glob('./imgs/*.pkl')
            for file in pkl_paths:
                os.remove(file)
        else:
            for item in result['identity']:
                os.remove(item)
            pkl_paths = glob.glob('./imgs/*.pkl')
            for file in pkl_paths:
                os.remove(file)

    generate_embeddings()















