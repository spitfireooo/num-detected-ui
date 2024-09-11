import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NumbDetected:
    def __init__(self, image, epochs, validation_split):
        (x_train, y_train), (x_test, y_test) = self.load_dataset()
        self.y_train_cat, self.y_test_cat = self.output_values(y_train, y_test)
        self.x_train, self.y_train = self.normilize_data(x_train, y_train)
        self.model = self.train(
            self.ai_model(28, 28, 1, 128, 10),
            self.x_train,
            self.y_train_cat,
            x_test,
            self.y_test_cat,
            epochs,
            validation_split
        )
        self.img = self.custom_image(image)
        self.res = self.result(self.img, self.model)
        self.drawer(self.img)


    def ai_model(self, width, height, depth, relu, softmax):
        return self.optimize(keras.Sequential([
            Flatten(input_shape=(width, height, depth)),
            Dense(relu, activation='relu'),
            Dense(softmax, activation='softmax')
        ]))

    def train(self, model, x_train, y_train_cat, x_test, y_test_cat, epochs=5, validation_split=0.2):
        if validation_split > 0.5 or validation_split < 0.1:
            validation_split = 0.2
        model.fit(x_train, y_train_cat, batch_size=32, epochs=epochs, validation_split=validation_split)
        model.evaluate(x_test, y_test_cat)
        return model

    def optimize(self, model):
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_dataset(self):
        return mnist.load_data()

    def normilize_data(self, x_train, x_test):
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return x_train, x_test

    def output_values(self, y_train, y_test):
        y_train_cat = keras.utils.to_categorical(y_train, 10)
        y_test_cat = keras.utils.to_categorical(y_test, 10)
        return y_train_cat, y_test_cat

    def custom_image(self, img="ai.jpeg"):
        print("IMG", img)
        load_image = plt.imread(img, format="jpeg")
        gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        image = 1 - (gray(load_image).astype("float32") / 255)
        return image

    def result(self, image, model):
        x = np.expand_dims(image, axis=0)
        res = model.predict(x)
        print(res)
        print(f"Распознанная цифра: {np.argmax(res)}")
        return res

    def drawer(self, image):
        plt.imshow(image, cmap=plt.cm.binary)
        plt.show()



if __name__ == '__main__':
    image = str(input("Filename: "))
    epochs = int(input("Epochs: "))
    validation_split = float(input("Validation split: "))
    NumbDetected(image, epochs, validation_split)