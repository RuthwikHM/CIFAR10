#Importing libraries
import numpy as np
import keras
#Importing dataset,required layers and models
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
num_classes=10
print('Loading data\n')
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
print('Training sample')
plt.imshow(x_train[0])
plt.show()
print('Test sample')
plt.imshow(x_test[0])
plt.show()
#Converting class vectors to binary class matrices
y_train=to_categorical(y_train,num_classes)
y_test=to_categorical(y_test,num_classes)
#Creating the model
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
#Compiling the model
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_train/=255
print('Fitting data to the model')
#Fitting data to model
model.fit(x_train,y_train,batch_size=32,epochs=25,validation_split=0.2)
print('Evaluating the test data on model')
score=model.evaluate(x_test,y_test,batch_size=32)
print('Test accuracy=',score[1])
