import cv2
import numpy as np
import tensorflow as tf

trainside = tf.keras.utils.image_dataset_from_directory(
                                            "/Users/lait-zet/Desktop/Facial-Emotion-Detection/data/train",
                                             shuffle=True,
                                             batch_size=10,
                                             image_size=(48,48),
                                             validation_split=0.2,
                                             subset='training',
                                             seed = 42
                                             )
validationdata = tf.keras.utils.image_dataset_from_directory(
                                            "/Users/lait-zet/Desktop/Facial-Emotion-Detection/data/train",
                                             shuffle=True,
                                             batch_size=10,
                                             image_size=(48,48),
                                             validation_split=0.2,
                                             subset='validation',
                                             seed = 42
                                             )


class_names = trainside.class_names


cascPath = '/Users/lait-zet/Desktop/Facial-Emotion-Detection/haarscad_Propgram/haarcascade_frontalface_default 2.xml'


faceCascade = cv2.CascadeClassifier(cascPath)

predictor_model = tf.keras.models.load_model('/Users/lait-zet/Desktop/Facial-Emotion-Detection/My_Model/emotion_model.h5')



def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0),    2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        roi_color = cv2.resize(roi_color, (48, 48))
        roi_color = roi_color / 255.0
        roi_color = np.expand_dims(roi_color, axis=0)
        prediction = predictor_model.predict(roi_color)
        maxindex = int(np.argmax(prediction))
        predicted_emotion = class_names[maxindex]
        confidence = float(np.max(prediction))

        return predicted_emotion, confidence


# image_path = '/Users/lait-zet/Desktop/Facial-Emotion-Detection/images_tester/image_test.webp'

# image = cv2.imread(image_path)


# predicted_emotion, confidence = predict_emotion(image)
# print(f"predicted emotion: {predicted_emotion}, confidence: {confidence}")
# cv2.putText(image, f"{predicted_emotion} ({confidence*100:.2f}%)", (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()