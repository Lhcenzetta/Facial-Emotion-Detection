import cv2
import numpy as np
import tensorflow as tf

cascPath = '/Users/lait-zet/Desktop/Facial-Emotion-Detection/haarscad_Propgram/haarcascade_frontalface_default 2.xml'


faceCascade = cv2.CascadeClassifier(cascPath)

predictor_model = tf.keras.models.load_model('/Users/lait-zet/Desktop/Facial-Emotion-Detection/My_Model/emotion_model.h5')

image_path = '/Users/lait-zet/Desktop/Facial-Emotion-Detection/images_tester/image_test.webp'

image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
    roi_color = cv2.resize(roi_color, (48, 48))
    roi_color = roi_color / 255.0
    roi_color = np.expand_dims(roi_color, axis=0)

    predictions = predictor_model.predict(roi_color)
    max_index = int(tf.argmax(predictions[0]))
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    predicted_emotion = emotions[max_index]

    cv2.putText(image, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()