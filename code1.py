import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("keras_model.h5")

video = cv2.VideoCapture(0)

while True: 
    check, frame = video.read()
    if check:
        img = cv2.resize(frame, (224,224))

        img2 = np.array(img, dtype = np.float32)
        img2 = np.expand_dims(img2, axis = 0)
        img_convertida = img2/255.0

        cv2.imshow("Resultado", frame)

        previsao = model.predict(img_convertida)
        print(previsao)

    if cv2.waitKey(1) == 32:
        break