import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import time
import pickle

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

st = time.time()
### load data
# train = pd.read_csv("C://Users/edwardchen/.kaggle/competitions/digit-recognizer/train.csv")
train = pd.read_csv("/app/Data/train.csv")
train = train.sample(frac=0.05, replace=False) # 用小樣本測試
# test = pd.read_csv("C://Users/edwardchen/.kaggle/competitions/digit-recognizer/test.csv")
test = pd.read_csv("/app/Data/test.csv")

### Preprocess
# sperate training label and data
Y_train = train['label']
X_train = train.drop(labels = 'label', axis=1)

# check label fraction
#sns.countplot(Y_train)
#plt.show() # can see that each label is around 4000 observations

# check if missing value
# print(X_train.isnull().any().describe()) # 我的做法 sum(X_train.isnull()) 看True加起來是多少

# normalization    gray-sacle 0~255 ---> 0~1    Why?: 1. Reduce illumination difference. 2. 0~1 makes CNN converge faster than 0~255. 
# **************
# Q. How about 0~255 -> 0~0.01? more smaller than 0~1
# **************
X_train = X_train/255.0
test = test/255.0

# Reshape 784 -> 28*28*1 (height*width*channel)
X_train = X_train.values.reshape(-1,28,28,1) # 第一個維度 -1，表示 record 數量不做更動，每個records改為 28*28*1的 3D shape
test = test.values.reshape(-1,28,28,1)

# Label encoding
Y_train = to_categorical(Y_train, num_classes=10)
# **************
# Q. How about not doing this?
# **************

# Split training and validation set
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

# Show one observation
# g = plt.imshow(X_train[0][:,:,0])
# plt.show()

### CNN model
# Define model
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters = 32, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Conv2D(filters = 64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Set the optimizer (由optimizer迭代更新 filters kernel values, weights, bias of each neurons)
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
print('lr reduction: {}'.format(learning_rate_reduction))
epochs = 1 
batch_size = 86

# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10, # degrees 0 ~ 180.
    zoom_range=0.1, # randomly zoom by 10%.
    width_shift_range=0.1, # randomly shift image horizontally by 10% of the width.
    height_shift_range=0.1, # randomly shift image vertically by 10% heigth.
    horizontal_flip=False,
    vertical_flip=False
)
datagen.fit(X_train)

# Fit the model
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, Y_val),
    verbose=2,
    steps_per_epoch=X_train.shape[0],
    callbacks=[learning_rate_reduction])

# 保存 模型
file = open('history.pickle', 'wb')
pickle.dump(history, file)
file.close()

### Evaluation the model
# training and validaion curves
# fig, ax = plt.subplot(2,1) # draw 2 plots (Loss and Accuracy)
# ax[0].plot(history.history['loss'], col='b', label='Training Loss')
# ax[0].plot(history.history['val_loss'], col='r', label='Validation Loss', axes=ax[0])
# legend = ax[0].legend(loc='best', shadow=True)

# ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
# ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
# legend = ax[1].legend(loc='best', shadow=True)
# plt.show()

# confusion matrix
y_true = np.argmax(Y_val, axis=1)   # argmax return 陣列中 數字最大處的 index。這裡是回傳softmax結果數值最高的位置，即 0~9
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
print('confusion maxtirx:')
print(confusion_mtx)

### Predict and Submit
results = model.predict(test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)
submission.to_csv('digit-recognizer_cnn_submit.csv', index=False)

ft = time.time()
print('time:{}'.format(ft-st))