import os
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, MaxPool2D, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import json
from sklearn.metrics import classification_report, precision_score
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image



anapath = '/gdrive/My Drive/YAPAY ZEKA/archive'
anapath2 = '/gdrive/My Drive/YAPAY ZEKA/Duygu Tanıma Bitirme Projesi'
path = 'archive/train'
path2 = 'archive/test'

train_img = []
train_sinif = []

test_img = []
test_sinif = []

     
def model_save(model, isim):
      model.save(f"{isim}.h5")

def hist_save(hist,isim):
      with open(f'{isim}', 'wb') as f:
        pickle.dump(hist.history, f)

def model_load(isim):
      return load_model(f"{isim}" + ".h5")
    
def hist_load(hist):
      with open(f"{hist}", 'rb') as f:
        return pickle.load(f)      

def oku(path_, list1, list2):
    list1 = list1
    list2 = list2
    for i in range(7):
        if i in (0,3,5):
            myImageList = os.listdir(path_ + "/"+str(i))      
            print(i)
            for j in myImageList:
                img = cv2.imread(path_ + "/" + str(i) + "/" + str(j))
                img = cv2.resize(img, (48,48))
                list1.append(img)
                list2.append(i)
    return list1, list2

def train_oku():
  oku(path, train_img, train_sinif)

def test_oku():
  oku(path2, test_img, test_sinif)
  
def bolme():
    x_train, x_validation, y_train, y_validation = train_test_split(train_img, train_sinif, test_size = 0.1, random_state = 42)
    return x_train, y_train, x_validation, y_validation
  
 
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img /255
    
    return img 
  
train_oku()
test_oku()


plt.imshow(train_img[1])
plt.show()


x_train, y_train, x_validation, y_validation = bolme()

x_test = test_img
y_test = test_sinif


x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))


x_train = x_train.reshape(-1,48,48,1)
print(x_train.shape)

dataGen = ImageDataGenerator(width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             zoom_range = 0.1,
                             rotation_range = 10,
                             fill_mode = "nearest")


dataGen.fit(x_train)


for i in range(len(y_train)):
  if y_train[i] == 3:
    y_train[i] = 1
  elif y_train[i] == 5:
    y_train[i] = 2
  else:
    pass


for i in range(len(y_test)):
  if y_test[i] == 3:
    y_test[i] = 1
  elif y_test[i] == 5:
    y_test[i] = 2
  else:
    pass


for i in range(len(y_validation)):
  if y_validation[i] == 3:
    y_validation[i] = 1
  elif y_validation[i] == 5:
    y_validation[i] = 2
  else:
    pass

y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)
y_validation = to_categorical(y_validation, 3)




batch_size = 250

#CNN MODELi1
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", input_shape=(48,48,1)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])


#CNN MODELi2
model2 = Sequential()

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", input_shape=(48,48,1)))
model2.add(BatchNormalization())

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Flatten())
model2.add(Dense(1024, activation='relu'))
model2.add(Dropout(0.5))

model2.add(Dense(3, activation='softmax'))

model2.compile(loss = "categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])


#CNN MODELi3
model3 = Sequential()

model3.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", input_shape=(48,48,1)))
model3.add(BatchNormalization())

model3.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.5))

model3.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.5))

model3.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.5))

model3.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.5))

model3.add(Flatten())
model3.add(Dense(1024, activation='relu'))
model3.add(Dropout(0.5))

model3.add(Dense(3, activation='softmax'))

model3.compile(loss = "categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])


#CNN MODELi4
model4 = Sequential()

model4.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", input_shape=(48,48,1)))

model4.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.5))

model4.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.5))

model4.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.5))

model4.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.5))

model4.add(Flatten())
model4.add(Dense(1024, activation='relu'))
model4.add(Dropout(0.5))

model4.add(Dense(3, activation='softmax'))

model4.compile(loss = "categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])


callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, mode='max')


hist1 = model.fit(x_train, y_train, batch_size = batch_size, 
                                        validation_data = (x_validation, y_validation),
                                        epochs = 30, shuffle = 1, callbacks = [callback])

model_save(model, "hist1")
hist_save(hist1, "hist1")

hist2 = model.fit(dataGen.flow(x_train, y_train, batch_size = batch_size), 
                                        validation_data = (x_validation, y_validation),
                                        epochs = 30, shuffle = 1, callbacks = [callback])

model_save(model, "hist2")
hist_save(hist2, "hist2")

hist3 = model4.fit(x_train, y_train, batch_size = batch_size, 
                                        validation_data = (x_validation, y_validation),
                                        epochs = 30, shuffle = 1, callbacks = [callback])

model_save(model4, "hist3")
hist_save(hist3, "hist3")

hist4 = model4.fit_generator(dataGen.flow(x_train, y_train, batch_size = batch_size), 
                                        validation_data = (x_validation, y_validation),
                                        epochs = 30, shuffle = 1, callbacks = [callback])


plt.plot(hist1.history["loss"], label = "train loss")
plt.plot(hist1.history["val_loss"], label = "val loss")
plt.legend()
plt.show()

#plt.figure()
plt.plot(hist1.history["accuracy"], label = "train acc")
plt.plot(hist1.history["val_accuracy"], label = "val acc")
plt.legend()
plt.show()

plt.plot(hist2.history["loss"], label = "train loss")
plt.plot(hist2.history["val_loss"], label = "val loss")
plt.legend()
plt.show()

#plt.figure()
plt.plot(hist2.history["accuracy"], label = "train acc")
plt.plot(hist2.history["val_accuracy"], label = "val acc")
plt.legend()
plt.show()

plt.plot(hist3.history["loss"], label = "train loss")
plt.plot(hist3.history["val_loss"], label = "val loss")
plt.legend()
plt.show()

#plt.figure()
plt.plot(hist3.history["accuracy"], label = "train acc")
plt.plot(hist3.history["val_accuracy"], label = "val acc")
plt.legend()
plt.show()

plt.plot(hist4.history["loss"], label = "train loss")
plt.plot(hist4.history["val_loss"], label = "val loss")
plt.legend()
plt.show()

#plt.figure()
plt.plot(hist4.history["accuracy"], label = "train acc")
plt.plot(hist4.history["val_accuracy"], label = "val acc")
plt.legend()
plt.show()

load_model = load_model('hist2.h5')

y_pred = load_model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_test, axis = 1)
cm = confusion_matrix(y_true, y_pred_class)
f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot = True, linewidths = 0.01, cmap = "Reds", linecolor = "gray", fmt = ".1f", ax=ax)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("cm")
plt.show()

print(classification_report(y_true, y_pred_class))