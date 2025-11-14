# Détection des Émotions Faciales

Une application basée sur l'apprentissage automatique qui détecte et classifie les émotions humaines à partir d'images faciales en utilisant le deep learning et OpenCV. Ce projet combine des techniques de vision par ordinateur avec un backend FastAPI pour fournir une prédiction d'émotion en temps réel et un suivi historique.

## 🎯 Caractéristiques

- **Détection d'Émotion en Temps Réel**: Analyse les images faciales et prédit l'une des 7 émotions
- **Détection de Visages**: Utilise les classificateurs Haar Cascade pour une détection robuste des visages
- **API REST**: Points de terminaison basés sur FastAPI pour une intégration facile
- **Historique des Prédictions**: Stocker et récupérer les dossiers de prédiction avec persistance de la base de données
- **Catégories d'Émotions Multiples**: Détecte 7 émotions:
  - Colère
  - Dégoût
  - Peur
  - Joie
  - Neutre
  - Tristesse
  - Surprise

## 📁 Structure du Projet

```
Facial-Emotion-Detection/
├── app/                              # Code principal de l'application
│   ├── main.py                       # Application FastAPI et points de terminaison
│   ├── detect_and_predict.py        # Logique de détection et prédiction d'émotion
│   ├── models.py                     # Modèles de base de données SQLAlchemy
│   ├── schemas.py                    # Schémas de validation Pydantic
│   ├── database.py                   # Configuration de la base de données
│   └── __pycache__/
├── data/                             # Ensembles de données d'entraînement et de test
│   ├── train/                        # Images d'entraînement organisées par émotion
│   │   ├── angry/
│   │   ├── disgusted/
│   │   ├── fearful/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprised/
│   └── test/                         # Images de test organisées par émotion
├── haarscad_Propgram/                # Fichiers du classificateur Haar Cascade
│   └── haarcascade_frontalface_default 2.xml
├── My_Model/                         # Modèle de détection d'émotion entraîné
│   └── emotion_detection_model.pkl
├── test_u/                           # Tests unitaires
│   └── test_project.py
├── EDA.ipynb                         # Cahier d'Analyse Exploratoire des Données
├── images_tester/                    # Exemples d'images pour les tests
└── README.md
```

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.8+
- Gestionnaire de paquets pip ou conda
- OpenCV
- TensorFlow/Keras
- FastAPI
- SQLAlchemy
- joblib

### Installation

1. Cloner le référentiel:
```bash
git clone https://github.com/Lhcenzetta/Facial-Emotion-Detection.git
cd Facial-Emotion-Detection
```

2. Créer et activer un environnement virtuel:
```bash
python -m venv venv
source venv/bin/activate  # Sous Windows: venv\Scripts\activate
```

3. Installer les dépendances requises:
```bash
pip install -r requirements.txt
```

4. Télécharger les données:

Les données d'entraînement et de test peuvent être téléchargées depuis Kaggle:
```
https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data
```

Après téléchargement, extrayez les fichiers dans le dossier `data/` du projet.

### Exécution de l'Application

Démarrez le serveur FastAPI:
```bash
uvicorn app.main:app --reload
```

L'API sera disponible à `http://localhost:8000`

La documentation de l'API est disponible à:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📡 Points de Terminaison API

### 1. Prédire une Émotion
**POST** `/predict_emotion`

Téléchargez un fichier image pour prédire l'émotion des visages détectés.

**Demande:**
- `file`: Fichier image (données de formulaire multipart)

**Réponse:**
```json
{
  "id": 1,
  "emotion": "joie",
  "score": 0.95,
  "create_at_date": "2025-11-14 10:30:45"
}
```

### 2. Obtenir l'Historique des Prédictions
**GET** `/history`

Récupérez toutes les prédictions stockées.

**Réponse:**
```json
[
  {
    "id": 1,
    "emotion": "joie",
    "score": 0.95,
    "create_at_date": "2025-11-14 10:30:45"
  },
  ...
]
```

### 3. Obtenir une Prédiction Spécifique
**GET** `/history/{prediction_id}`

Récupérez une prédiction spécifique par ID.

**Réponse:**
```json
{
  "id": 1,
  "emotion": "joie",
  "score": 0.95,
  "create_at_date": "2025-11-14 10:30:45"
}
```

## 🤖 Comment ça Marche

1. **Détection de Visages**: L'application utilise le Classificateur Haar Cascade pour détecter les visages dans l'image d'entrée
2. **Prétraitement**: Les régions de visage détectées sont converties en niveaux de gris et redimensionnées à 48x48 pixels
3. **Prédiction d'Émotion**: Le visage prétraité est introduit dans un modèle de deep learning entraîné (stocké sous le nom `emotion_detection_model.pkl`)
4. **Notation**: Le modèle renvoie la classe d'émotion prédite et le score de confiance
5. **Stockage**: Les résultats sont stockés dans une base de données SQLite pour un suivi historique

## 📊 Détails du Modèle

- **Type de Modèle**: Réseau de neurones pré-entraîné (sauvegardé en tant que fichier pickle joblib)
- **Taille d'Entrée**: Images en niveaux de gris 48x48
- **Classes de Sortie**: 7 catégories d'émotions
- **Détecteur de Visages**: Classificateur Haar Cascade OpenCV

## 🧪 Tests

Exécutez la suite de tests:
```bash
pytest test_u/test_project.py
```

## 📓 Analyse Exploratoire des Données

Pour une analyse détaillée de l'ensemble de données, consultez `EDA.ipynb` qui comprend:
- Analyse de la distribution de l'ensemble de données
- Techniques de prétraitement d'images
- Aperçus de la formation du modèle

## 🛠️ Technologies Utilisées

- **FastAPI**: Framework web moderne pour construire des API
- **TensorFlow/Keras**: Framework de deep learning
- **OpenCV**: Bibliothèque de vision par ordinateur
- **SQLAlchemy**: ORM pour les opérations de base de données
- **Pydantic**: Validation et analyse des données
- **SQLite**: Base de données pour stocker les prédictions
- **Joblib**: Sérialisation et chargement de modèles

## 📝 Schéma de la Base de Données

### Tableau des Prédictions

| Colonne | Type | Description |
|---------|------|-------------|
| id | Integer | Clé primaire |
| emotion | String | Classe d'émotion prédite |
| score | Float | Score de confiance (0-1) |
| create_at_date | String | Horodatage de la prédiction |

## 🤝 Contribution

Les contributions sont bienvenues! N'hésitez pas à soumettre une demande d'extraction.

## 📄 Licence

Ce projet est open source et disponible sous la licence MIT.

## 👨‍💻 Auteur

- **Lhcenzetta** - [Profil GitHub](https://github.com/Lhcenzetta)

## 🙏 Remerciements

- Classificateurs Haar Cascade d'OpenCV
- Inspiration de la recherche en reconnaissance des émotions
- Contributions et retours de la communauté