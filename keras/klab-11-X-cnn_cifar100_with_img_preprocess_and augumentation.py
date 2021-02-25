from keras.datasets import cifar100
from keras.models import *
from keras.layers import *
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import *
from keras.preprocessing import image
from keras import regularizers,optimizers
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
 
def lr_schedule(epoch):
    lrate = 0.001
    if epoch < 2:
        lrate = 0.005
    if epoch > 5:
        lrate = 0.0001
    return lrate
 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
#c
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#image augumentation
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator( rotation_range=30,
                 width_shift_range=0.1, height_shift_range=0.1,
                 horizontal_flip=True)
datagen.fit(x_train)

#cofirm train data
def show_imgs(X):
    plt.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            plt.subplot2grid((4,4),(i,j))
            plt.imshow((X[k]))
            k = k+1
    # show the plot
    plt.show()
show_imgs(x_train[:16])


#modeling
model  = Sequential()
model.add(Conv2D(32,(2,2),padding='same',input_shape=(32,32,3),activation='elu'))
model.add(MaxPool2D(2,2))    
model.add(Conv2D(64, (2, 2), padding='same',activation='elu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100,activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.rmsprop(lr=0.001, decay=0.00005), metrics=['accuracy'])

#model structure
print(model.summary())

callbacks = [EarlyStopping(monitor='val_loss', patience=1, mode='min', verbose=1)]

model.fit(x_train, y_train,epochs=10,validation_split=0.2,verbose=1,callbacks=callbacks)

res_acc = model.evaluate(x_test,y_test)

#res_acc: 0.9901830131530762
print("res_acc:",res_acc[1])
#res_acc: 0.043311127352714536
print("res_socre:",res_acc[0])