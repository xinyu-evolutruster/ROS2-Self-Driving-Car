import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf

TRAIN_CNN = True
EVALUATE_MODEL = True

ROOT = "/home/cindy/ROS2-Self-Driving-Car/prius_sdc_package/prius_sdc_package"

NUM_CATEGORIES = 0
sign_classes = ["speed_sign_30", "speed_sign_60", "speed_sign_90", "stop", "left_turn", "no_sign"]

def load_data(data_dir):
    '''
    Loading data from Train folder.
    
    Returns a tuple `(images, labels)` , where `images` is a list of all the images in the train directory,
    where each image is formatted as a numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. 
    `labels` is a list of integer labels, representing the categories for each of the
    corresponding `images`.
    '''
    global NUM_CATEGORIES
    images = list()
    labels = list()

    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        for img in os.listdir(category_path):
            img = load_img(os.path.join(category_path, img), target_size=(30,30))
            image = img_to_array(img)
            images.append(image)
            labels.append(category)

    return images, labels

def train_signs_model(data_dir, IMG_HEIGHT=30, IMG_WIDTH=30, EPOCHS=30, save_model=True, saved_model="saved_model_5_sign.h5"):
    train_path = data_dir + "/datasets"

    global NUM_CATEGORIES
    NUM_CATEGORIES = len(os.listdir(train_path))
    print("NUM_CATEGORIES = ", NUM_CATEGORIES)

    # visualizing all the different signs
    img_dir = pathlib.Path(train_path)
    plt.figure(figsize=(14,14))
    index = 0
    for i in range(NUM_CATEGORIES):
        plt.subplot(7, 7, i+1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        # print(img_dir)
        sign = list(img_dir.glob(f'{i}/*'))[0]
        img = load_img(sign, target_size=(IMG_WIDTH, IMG_HEIGHT))
        plt.imshow(img)
    plt.show()

    images, labels = load_data(train_path)
    print(len(labels))
    # one hot encoding the labels
    labels = to_categorical(labels)

    # splitting the dataset into training and test set
    x_train, x_test, y_train, y_test = train_test_split(np.array(images), labels, test_size=0.4)

    # ================== model creation ====================
    model = Sequential()
    # first convolutional layer
    model.add(Conv2D(filters=16, kernel_size=3, activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    # second convolutional layer
    model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    # flattening the layer and adding dense layer
    model.add(Flatten())
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(NUM_CATEGORIES, activation="softmax"))
    model.summary()
    # ======================================================

    # compiling the model
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # fitting the model
    history = model.fit(x_train, 
                        y_train,
                        validation_data = (x_test, y_test),
                        epochs=EPOCHS,
                        steps_per_epoch=60)
    
    loss, accuracy = model.evaluate(x_test, y_test)
    print("test set accuracy: ", accuracy * 100)

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label="Training Accuracy")
    plt.plot(epochs_range, val_accuracy, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validating Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()

    # ================== saving model ====================
    # save model and architecture to a single file
    saved_model_path = os.path.join(data_dir, saved_model)
    if save_model:
        model.save(saved_model_path)
        print("saved model to disk")
    # ====================================================

def evaluate_model_on_image(data_dir, image_path="", image_label=None):
    # load model
    model_path = os.path.join(data_dir, "saved_model_5_sign.h5")
    model = load_model(model_path)
    # summarize model
    model.summary()
    # load dataset
    if image_path == "":
        def_test_images_path = os.path.join(data_dir, "test_images")
        # number of classes
        global NUM_CATEGORIES
        NUM_CATEGORIES = len(os.listdir(def_test_images_path))
        print("NUM_CATEGORIES = ", NUM_CATEGORIES)

        # visualizing all the different signs
        img_dir = pathlib.Path(def_test_images_path)
        plt.figure(figsize=(14,14))
        for i in range(NUM_CATEGORIES):
            ax = plt.subplot(1, 6, i+1)
            plt.tight_layout()
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            sign = list(img_dir.glob(f'{i}/*'))[0]
            img = load_img(sign, target_size=(30, 30))
            image = img_to_array(img)
            images = list()
            images.append(image)
            plt.imshow(img)
            y_pred = model.predict(np.array(images))
            actual_vs_predicted = "pred = " + sign_classes[np.argmax(y_pred)] + "\nActual = " + sign_classes[i] + "\n"
            plt.xlabel(actual_vs_predicted)
        plt.show()
    else:
        # split into input(X) and output(Y) variables
        output = []
        image = load_img(image_path, target_size=(30, 30))
        output.append(np.array(image))
        x_test = np.array(output)

        X = np.array(image).reshape(1, 30, 30, 3)

        Y = np.array([[0, 0, 0, 0]])
        Y[0, image_label] = 1

        print("X shape is: {}".format(X.shape))
        print("Y shape is: {}".format(Y.shape))
        # evaluate the model
        score = model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_name[1], score[1] * 100))

def main():
    data_dir = os.path.join(ROOT, "data")
    if TRAIN_CNN:
        train_signs_model(data_dir)
    if EVALUATE_MODEL:
        evaluate_model_on_image(data_dir)

if __name__ == '__main__':
    main()