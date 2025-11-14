import joblib
import pytest
import os

@pytest.fixture
def load_model():
    # Get repo root dynamically
    repo_root = os.path.dirname(os.path.dirname(__file__))

    # Relative path inside the repo
    model_path = os.path.join(repo_root, "My_Model/emotion_detection_model.pkl")

    # Load model
    model = joblib.load(model_path)
    return model

def test_model_loading(load_model):
    assert load_model is not None
    