import datetime
from fastapi import FastAPI, Depends, UploadFile, File
from sqlalchemy.orm import Session
import numpy as np
import cv2
import models
import schemas
from database import engine, get_db
from detect_and_predict import predict_emotion 


models.Base.metadata.create_all(bind=engine)


app = FastAPI()


@app.post("/predict_emotion", response_model=schemas.PredictionBase)
def upload(file : UploadFile = File(...) ,db: Session = Depends(get_db)):
    image_data = file.file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    emotion, score = predict_emotion(image)

    new_data = {
        "emotion": emotion,
        "score": score,
        "create_at_date" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    prediction = models.Prediction(**new_data)
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    
    return prediction

@app.get("/history", response_model=list[schemas.Prediction])
def get_history(db: Session = Depends(get_db)):
    predictions = db.query(models.Prediction).all()
    return predictions

@app.get("/history/{prediction_id}", response_model=schemas.Prediction)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    prediction = db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()
    return prediction
