# Facial Emotion Detection

A machine learning-based application that detects and classifies human emotions from facial images using deep learning and OpenCV. This project combines computer vision techniques with a FastAPI backend to provide real-time emotion prediction and historical tracking.

## 🎯 Features

- **Real-time Emotion Detection**: Analyzes facial images and predicts one of 7 emotions
- **Face Detection**: Uses Haar Cascade classifiers for robust face detection
- **REST API**: FastAPI-based endpoints for easy integration
- **Prediction History**: Store and retrieve prediction records with database persistence
- **Multiple Emotion Categories**: Detects 7 emotions:
  - Angry
  - Disgusted
  - Fearful
  - Happy
  - Neutral
  - Sad
  - Surprised

## 📁 Project Structure

```
Facial-Emotion-Detection/
├── app/                              # Main application code
│   ├── main.py                       # FastAPI application and endpoints
│   ├── detect_and_predict.py        # Emotion detection and prediction logic
│   ├── models.py                     # SQLAlchemy database models
│   ├── schemas.py                    # Pydantic validation schemas
│   ├── database.py                   # Database configuration
│   └── __pycache__/
├── data/                             # Training and test datasets
│   ├── train/                        # Training images organized by emotion
│   │   ├── angry/
│   │   ├── disgusted/
│   │   ├── fearful/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprised/
│   └── test/                         # Test images organized by emotion
├── haarscad_Propgram/                # Haar Cascade classifier files
│   └── haarcascade_frontalface_default 2.xml
├── My_Model/                         # Trained emotion detection model
│   └── emotion_detection_model.pkl
├── test_u/                           # Unit tests
│   └── test_project.py
├── EDA.ipynb                         # Exploratory Data Analysis notebook
├── images_tester/                    # Sample images for testing
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager
- OpenCV
- TensorFlow/Keras
- FastAPI
- SQLAlchemy
- joblib

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Lhcenzetta/Facial-Emotion-Detection.git
cd Facial-Emotion-Detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

API documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📡 API Endpoints

### 1. Predict Emotion
**POST** `/predict_emotion`

Upload an image file to predict the emotion of detected faces.

**Request:**
- `file`: Image file (multipart form data)

**Response:**
```json
{
  "id": 1,
  "emotion": "happy",
  "score": 0.95,
  "create_at_date": "2025-11-14 10:30:45"
}
```

### 2. Get Prediction History
**GET** `/history`

Retrieve all stored predictions.

**Response:**
```json
[
  {
    "id": 1,
    "emotion": "happy",
    "score": 0.95,
    "create_at_date": "2025-11-14 10:30:45"
  },
  ...
]
```

### 3. Get Specific Prediction
**GET** `/history/{prediction_id}`

Retrieve a specific prediction by ID.

**Response:**
```json
{
  "id": 1,
  "emotion": "happy",
  "score": 0.95,
  "create_at_date": "2025-11-14 10:30:45"
}
```

## 🤖 How It Works

1. **Face Detection**: The application uses Haar Cascade Classifier to detect faces in the input image
2. **Preprocessing**: Detected face regions are converted to grayscale and resized to 48x48 pixels
3. **Emotion Prediction**: The preprocessed face is fed into a trained deep learning model (stored as `emotion_detection_model.pkl`)
4. **Scoring**: The model returns the predicted emotion class and confidence score
5. **Storage**: Results are stored in a SQLite database for historical tracking

## 📊 Model Details

- **Model Type**: Pre-trained neural network (saved as joblib pickle file)
- **Input Size**: 48x48 grayscale images
- **Output Classes**: 7 emotion categories
- **Face Detector**: OpenCV Haar Cascade Classifier

## 🧪 Testing

Run the test suite:
```bash
pytest test_u/test_project.py
```

## 📓 Exploratory Data Analysis

For detailed analysis of the dataset, see `EDA.ipynb` which includes:
- Dataset distribution analysis
- Image preprocessing techniques
- Model training insights

## 🛠️ Technologies Used

- **FastAPI**: Modern web framework for building APIs
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision library
- **SQLAlchemy**: ORM for database operations
- **Pydantic**: Data validation and parsing
- **SQLite**: Database for storing predictions
- **Joblib**: Model serialization and loading

## 📝 Database Schema

### Predictions Table

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| emotion | String | Predicted emotion class |
| score | Float | Confidence score (0-1) |
| create_at_date | String | Timestamp of prediction |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.


## 🙏 Acknowledgments

- Haar Cascade classifiers from OpenCV
- Inspiration from emotion recognition research
- Community contributions and feedback