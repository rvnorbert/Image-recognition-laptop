import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')

img_width, img_height = 512, 512
batch_size = 4
class_names = ["Green Light", "Red Light", "Yellow Light"]


def datagen():
    data = keras.utils.image_dataset_from_directory(
        'data',
        labels='inferred',
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="training"
    )

    data = data.map(lambda x, y: (x/255, y))
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    print(len(data))

    model = keras.models.Sequential()
    model.add(Input(shape=(img_width, img_height, 3)))
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #print(model.summary())

    print(len(data))

    train_size = int(len(data) * 0.8)
    validation_size = int(len(data) * 0.1) + 1
    test_size = int(len(data) * 0.1)

    #print(train_size)
    #print(validation_size)
    #print(test_size)

    train_dataset = data.take(train_size)
    validation_dataset = data.skip(train_size).take(validation_size)
    test_dataset = data.skip(test_size+validation_size).take(test_size)

    hist = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    model.save('image_classification_model.keras')


def dataset_gen():

    """model = keras.models.Sequential()
    model.add(Input(shape=(img_width, img_height, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    ds_train = keras.preprocessing.image_dataset_from_directory(
        'data',
        labels='inferred',
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="training",
    )

    ds_validation = keras.preprocessing.image_dataset_from_directory(
        'data',
        labels='inferred',
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_width, img_height),
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="validation",
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(ds_train, epochs=7, validation_data=ds_validation)

    loss, accuracy = model.evaluate(ds_validation)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    model.save('image_classification_model.keras')"""


def test():
    model = keras.models.load_model('image_classification_model.keras')
    imgr1 = cv.imread('TestImages/red1.jpg')
    imgr1 = cv.cvtColor(imgr1, cv.COLOR_BGR2RGB)
    imgr1 = cv.resize(imgr1, (img_width, img_height))

    prediction = model.predict(np.array([imgr1/255]))
    index = np.argmax(prediction)
    print(f"Predicted1: {class_names[index]}, Accuracy: {prediction[0][index]}")

    imgr2 = cv.imread('TestImages/red2.jpg')
    imgr2 = cv.cvtColor(imgr2, cv.COLOR_BGR2RGB)
    imgr2 = cv.resize(imgr2, (img_width, img_height))

    prediction = model.predict(np.array([imgr2/255]))
    index = np.argmax(prediction)
    print(f"Predicted2: {class_names[index]}, Accuracy: {prediction[0][index]}")

    imgg1 = cv.imread('TestImages/green1.jpg')
    imgg1 = cv.cvtColor(imgg1, cv.COLOR_BGR2RGB)
    imgg1 = cv.resize(imgg1, (img_width, img_height))

    prediction = model.predict(np.array([imgg1/255]))
    index = np.argmax(prediction)
    print(f"Predicted3: {class_names[index]}, Accuracy: {prediction[0][index]}")

    imgg2 = cv.imread('TestImages/green2.jpg')
    imgg2 = cv.cvtColor(imgg2, cv.COLOR_BGR2RGB)
    imgg2 = cv.resize(imgg2, (img_width, img_height))

    prediction = model.predict(np.array([imgg2/255]))
    index = np.argmax(prediction)
    print(f"Predicted4: {class_names[index]}, Accuracy: {prediction[0][index]}")

    imgy1 = cv.imread('TestImages/yellow1.jpg')
    imgy1 = cv.cvtColor(imgy1, cv.COLOR_BGR2RGB)
    imgy1 = cv.resize(imgy1, (img_width, img_height))

    prediction = model.predict(np.array([imgy1/255]))
    index = np.argmax(prediction)
    print(f"Predicted5: {class_names[index]}, Accuracy: {prediction[0][index]}")

    imgy2 = cv.imread('TestImages/yellow2.jpg')
    imgy2 = cv.cvtColor(imgy2, cv.COLOR_BGR2RGB)
    imgy2 = cv.resize(imgy2, (img_width, img_height))

    prediction = model.predict(np.array([imgy2/255]))
    index = np.argmax(prediction)
    print(f"Predicted6: {class_names[index]}, Accuracy: {prediction[0][index]}")


if __name__ == '__main__':
    #dataset_gen()
    test()
    #datagen()

