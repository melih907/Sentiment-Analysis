from json import load
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
import os
import spoti

from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np


def on_resize(event):
    image = bgimg.resize((event.width, event.height), Image.ANTIALIAS)
    l.image = ImageTk.PhotoImage(image)
    l.config(image=l.image)
    
    
def preProcess(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img /255.0
        
        return img

def canli():
    
    cap = cv2.VideoCapture(0)
    cap.set(3,480)
    cap.set(4,480)

    while True:
        
        success, frame = cap.read()
        
        img = np.asarray(frame)
        img = cv2.resize(img, (48,48))
        img = preProcess(img)
        
        
        img = img.reshape(1,48,48,1)
        
        
        class_names = ['kizgin', 'mutlu', 'uzgun']
        y_pred = load_model.predict(img)
        sinif = class_names[np.argmax(y_pred)]
            
        cv2.putText(frame, str(sinif) + " " , (50,50),cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0),1)

        cv2.imshow("Duygu Tanıma",frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): break    
    


def tahmin(model,path):
    
    load_model = model

    image_path = path

    test_image = image.load_img(image_path, target_size=(48, 48), grayscale=True)

    test_data = image.img_to_array(test_image)
    
    test_data = test_data/255

    test_data = np.expand_dims(test_data, axis=0)

    test_data = np.vstack([test_data])

    results = load_model.predict(test_data, batch_size=1)

    class_names = ['kizgin', 'mutlu', 'uzgun']
    
    print("Sınıflandırma sonucu en yüksek oranla:", class_names[np.argmax(results)])
    
    return class_names[np.argmax(results)], load_model



def upload_file():
    global img
    f_types=[("jpeg files", "*.jpg"), ("png files", "*.png")]
    filename = filedialog.askopenfilename(filetypes=f_types)
    print(filename)
    img = ImageTk.PhotoImage(file=filename)
    b3 =tk.Button(frame,image=img)
    b3.grid(row=3,column=1)
    
    duygu, _ = tahmin(load_model, filename)
    
    if duygu == "mutlu":
        spoti.mutlu()

    else:
        spoti.uzgun()


def goruntu_al(path):
    global img
    img = ImageTk.PhotoImage(file=path)
    b4 =tk.Button(frame,image=img)
    b4.grid(row=3,column=1, ipadx=110, ipady = 80)
    
    duygu, _ = tahmin(load_model, path)
    
    if duygu == "mutlu":
        spoti.mutlu()
        
    else:
        spoti.uzgun()



def yuz_tanima():
    secenek = ""
    image_list = []
    face_cascade = cv2.CascadeClassifier("yuz.xml")
    cap = cv2.VideoCapture(0)
    i = 0
    img = ""
    x = 0
    y = 0
    w = 0
    h = 0
    while True:
        ret, frame = cap.read()
        if ret:
            face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)
            for (x,y,w,h) in face_rect:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = cv2.equalizeHist(img)
                img = img /255.0
                x = x
                y = y
                w = w
                h = h
                cv2.putText(frame, "Yuz Algilaniyor" ,(10,10),cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0),1)
            cv2.imshow("face detect", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): 
            secenek = ord("q")
            break
        
    cv2.imwrite("gorsel.jpg", img)
    path = str(os.getcwd())
    path = path.replace('\\','/') + "/gorsel.jpg"
    
    goruntu_al(path)
    
    cap.release()
    cv2.destroyAllWindows()


load_model = load_model("hist2.h5")

frame = tk.Tk()
frame.geometry('800x600')
my_font1=('times', 18, 'bold')

bgimg = Image.open('sanıser.png')
l = tk.Label(frame)
l.place(x=0, y=0, relwidth=1, relheight=1)
l.bind('<Configure>', on_resize)

l1 = tk.Label(frame,text='Duyguya Göre Şarkılar',width=40,font=my_font1)  
l1.grid(row=1,column=1)

b2 = tk.Button(frame, text='Fotoğraf Yükle', width=20,command = lambda:upload_file())
b2.grid(row=20,column=1) 

frame.mainloop()