import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import curses, time

def input_char(message):
    try:
        win = curses.initscr()
        win.addstr(0, 0, message)
        while True: 
            ch = win.getch()
            if ch in range(32, 127): 
                break
            time.sleep(0.05)
    finally:
        curses.endwin()
    return chr(ch)


model = tf.keras.models.load_model('mnist_model.h5')  # Load your trained model

image_number = 1
correct = 0
total = 0
unsure = 0

def Normalized(data):
    d = {i:0 for i in range(0,10)}
    for predicts in prediction:
        for i in range(len(predicts)):
             d[i] = round(predicts[i] / 1, 4) 
    print(d)

while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        img = np.invert(img)  # Invert pixel values
        img = img.reshape(1, 28, 28, 1)  # Reshape for model input
        prediction = model.predict(img)

        Normalized(prediction)

        predicted_digit = np.argmax(prediction)
        print("Predicted Digit:", predicted_digit)


        plt.imshow(img[0, :, :, 0], cmap=plt.cm.binary)  # Show the image
        plt.title(f"Predicted Digit: {predicted_digit} input Y/N")
        plt.show(block = False)
        plt.pause(0.05)
        plt.close()
        c = input_char('input Y if correct, N if incorrect, Other if Unsure')
        if c.lower() in ['y', 'Y']:
            correct += 1
        if c.lower() in ['y', 'Y', 'n','N']:
            total += 1
        else:
            unsure += 1

    except Exception as e:
        print("Error: ", e)
    finally:
        image_number += 1

try:
    print(f"Score: {correct}/{total} or", correct/total)
except:
    print("No Data")
print("Unsure:", unsure)
