import joblib
import pytest


@pytest.fixture
def load_model():
    model_path = "/Users/lait-zet/Desktop/Facial-Emotion-Detection/My_Model/emotion_detection_model.pkl"
    model = joblib.load(model_path)
    return model

def test_model_loading(load_model):
    assert load_model != None