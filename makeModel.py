import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from mnist import MNIST
from sklearn.model_selection import train_test_split
import time

# Input image dimensions
img_rows, img_cols = 28, 28

emnist_data = MNIST(path='/Users/katie/OneDrive/Desktop/gzip', return_type='numpy')
emnist_data.select_emnist('letters')
X, y = emnist_data.load_training()

# Model variables
batch_size = 128
num_classes = 26
epochs = 10

X = X.reshape(124800, 28, 28)
y = y.reshape(124800, 1)

y = y-1

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=111)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Rescale the image values to [0, 1]
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

### Set the CNN Architecture
##model = Sequential()
##model.add(Conv2D(32, kernel_size=(3, 3),
##              activation='relu',
##              input_shape=input_shape))
##model.add(Conv2D(64, (3, 3), activation='relu'))
##model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(Dropout(0.25))
##model.add(Flatten())
##model.add(Dense(128, activation='relu'))
##model.add(Dropout(0.5))
##model.add(Dense(num_classes, activation='softmax'))
##
### Comple the model
##model.compile(loss=keras.losses.categorical_crossentropy,
##           optimizer=keras.optimizers.Adadelta(),
##           metrics=['accuracy'])
##
##
##
### Train the model
##model.fit(x_train, y_train,
##       batch_size=batch_size,
##       epochs=epochs,
##       verbose=1,
##       validation_data=(x_test, y_test))
### Save the model weights for future reference
##model.save('emnist_cnn_model_katei.h5')

print('TESTS ON MAC MACHINE')

win_load_times = []
win_eval_times = []
mac_load_times = []
mac_eval_times = []
kt_load_times = []
kt_eval_times = []

for i in range(10):
    load_start = time.time()
    model = load_model('emnist_cnn_model_windows.h5')
    load_end = time.time()

    # Evaluate the model using Accuracy and Loss
    eval_start = time.time()
    score = model.evaluate(x_test, y_test, verbose=0)
    eval_end = time.time()
    print('Windows test loss:', score[0])
    print('Windows test accuracy:', score[1])

    print('Windows model load time:', load_end - load_start)
    print('Windows Model evaluate time:', eval_end - eval_start)

    win_load_times.append(load_end - load_start)
    win_eval_times.append(eval_end - eval_start)



    load_start = time.time()
    model = load_model('emnist_cnn_model_mac.h5')
    load_end = time.time()

    # Evaluate the model using Accuracy and Loss
    eval_start = time.time()
    score = model.evaluate(x_test, y_test, verbose=0)
    eval_end = time.time()
    print('Mac test loss:', score[0])
    print('Mac test accuracy:', score[1])

    print('Mac model load time:', load_end - load_start)
    print('Mac Model evaluate time:', eval_end - eval_start)

    mac_load_times.append(load_end - load_start)
    mac_eval_times.append(eval_end - eval_start)

    load_start = time.time()
    model = load_model('emnist_cnn_model_katei.h5')
    load_end = time.time()

    # Evaluate the model using Accuracy and Loss
    eval_start = time.time()
    score = model.evaluate(x_test, y_test, verbose=0)
    eval_end = time.time()
    print('Katei test loss:', score[0])
    print('Katei test accuracy:', score[1])

    print('Katei model load time:', load_end - load_start)
    print('Katei Model evaluate time:', eval_end - eval_start)

    kt_load_times.append(load_end - load_start)
    kt_eval_times.append(eval_end - eval_start)

print('Average load time of Windows Model:', sum(win_load_times) / len(win_load_times))
print('Average evaluation time of Windows Model:', sum(win_eval_times) / len(win_eval_times))

print('Average load time of Mac Model:', sum(mac_load_times) / len(mac_load_times))
print('Average evaluation time of Mac Model:', sum(mac_eval_times) / len(mac_eval_times))

print('Average load time of Katei Model:', sum(kt_load_times) / len(kt_load_times))
print('Average evaluation time of Katei Model:', sum(kt_eval_times) / len(kt_eval_times))
