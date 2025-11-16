# Disease Prediction Backend

This repository contains the backend implementation for the **HEAL-O** healthcare application. It provides intelligent disease prediction using machine learning, AI-powered medical suggestions via generative AI, and an Interactive Voice Response (IVR) system to improve accessibility for all users.

## Overview

The Disease Prediction Backend is a comprehensive healthcare solution that combines:
- **Machine Learning**: Predicts diseases based on user symptoms using trained ML models deployed on IBM Watson
- **Generative AI**: Provides conversational medical suggestions and recommendations using Google's Gemini AI via LangChain
- **IVR System**: Enables voice-based interaction through Twilio, making healthcare accessible to users with limited digital literacy or disabilities

## Features

### 🔬 ML-Based Disease Prediction
- Predicts diseases from a comprehensive list of 132 symptoms
- Uses one-hot encoding for symptom representation
- IBM Watson ML deployment for real-time predictions
- Supports fuzzy matching for symptom recognition

### 🤖 AI-Powered Medical Suggestions
- Context-aware conversational AI using Google Gemini
- Provides disease descriptions and treatment recommendations
- Maintains conversation history for personalized responses
- Multi-language support

### 📞 IVR Accessibility System
- Voice-based symptom input via phone calls
- Speech-to-text using Whisper AI
- Automated appointment booking
- SMS confirmations via Twilio
- Perfect for users with visual impairments or low digital literacy

### 🔐 Additional Features
- Firebase real-time database integration
- Email notifications for appointments
- RESTful API with FastAPI
- CORS-enabled for cross-origin requests

## Technology Stack

- **Framework**: FastAPI
- **ML Platform**: IBM Watson Machine Learning
- **AI/LLM**: Google Gemini (via LangChain)
- **Speech Recognition**: Whisper AI (Faster Whisper)
- **IVR**: Twilio Voice API
- **Database**: Firebase Realtime Database
- **NLP**: spaCy, NLTK
- **Other**: Python 3.x, scikit-learn, NumPy

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/praveen647/Disease-Prediction.git
cd Disease-Prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

5. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
```

## Configuration

Create a `.env` file in the root directory with the following environment variables:

### IBM Watson ML
```
API_KEY=your_ibm_watson_api_key
WATSON_URL=your_watson_ml_endpoint
```

### Google Gemini AI
```
GEMINI_API=your_google_gemini_api_key
```

### Firebase
```
FIREBASE_API=your_firebase_api_key
FIREBASE_AUTH=your_firebase_auth_domain
DB_URL=your_firebase_database_url
DATABASE_URL=your_firebase_database_url
FIREBASE_ID=your_firebase_project_id
STORAGE_BUCKET=your_firebase_storage_bucket
MESSAGING_ID=your_firebase_messaging_sender_id
APP_ID=your_firebase_app_id
MEASUREMENT_ID=your_firebase_measurement_id
```

### Twilio (for IVR)
```
SSID=your_twilio_account_sid
AUTH_TOKEN=your_twilio_auth_token
```

### Email
```
PASSWORD=your_email_app_password
```

## Usage

### Starting the Main API Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Starting the IVR Server

```bash
python IVR-MAIN.py
```

The IVR system will be available for incoming Twilio calls.

## API Endpoints

### Main API (`main.py`)

#### `GET /`
Health check endpoint
```json
{
  "message": "IBM BOT BASE URL"
}
```

#### `POST /create`
Predict disease from symptoms
```json
Request:
{
  "query": "I have fever, headache and body pain"
}

Response:
{
  "response": "AI-generated medical suggestion",
  "result": "Predicted disease name"
}
```

#### `POST /send`
Send appointment confirmation email
```json
Request:
{
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "1234567890",
  "preferredDate": "2024-01-15",
  "location": "Chennai",
  "description": "Follow-up consultation"
}
```

### IVR API (`IVR-MAIN.py`)

#### `POST /ivr`
Initiates voice call and records symptoms

#### `POST /process_recording`
Processes recorded audio and predicts disease

#### `POST /digit_gather`
Handles DTMF input for appointment booking

#### `POST /process_name_recording`
Records user name and sends SMS confirmation

## Project Structure

```
Disease-Prediction/
├── main.py                  # Main FastAPI application
├── IVR-MAIN.py             # Twilio IVR system
├── description.py          # Disease descriptions and prescriptions
├── fire_base.py           # Firebase database operations
├── model.ipynb            # ML model training notebook
├── Visualization.ipynb    # Data visualization notebook
├── label_encoder.pkl      # Trained label encoder
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Supported Diseases

The system can predict 41 different diseases including:
- Fungal infection
- Allergy, GERD
- Chronic cholestasis
- Drug Reaction
- Peptic ulcer disease
- AIDS, Diabetes
- Gastroenteritis
- Bronchial Asthma
- Hypertension, Migraine
- And many more...

For the complete list, see `description.py`.

## Supported Symptoms

The system recognizes 132 different symptoms including:
- Common symptoms: fever, cough, headache, fatigue
- Skin conditions: rash, itching, eruptions
- Respiratory: breathlessness, congestion, phlegm
- Digestive: nausea, vomiting, diarrhea
- And many more...

## Related Repositories

This backend is part of the **HEAL-O** healthcare ecosystem. For the frontend and complete application, please visit:
- Frontend Repository: [heal-o](https://github.com/praveen647/heal-o)

## How It Works

1. **Symptom Input**: Users describe their symptoms via text (API) or voice (IVR)
2. **Processing**: Symptoms are preprocessed, lemmatized, and converted to one-hot vectors
3. **Prediction**: ML model on IBM Watson predicts the disease
4. **AI Suggestions**: Gemini AI provides contextual medical advice
5. **Appointment**: Users can book appointments via voice or email
6. **Confirmation**: Automated email/SMS confirmations are sent

## Security Notes

- Never commit `.env` files or `serviceAccountKey.json` to version control
- Use environment variables for all sensitive credentials
- Ensure Firebase security rules are properly configured
- Keep API keys secure and rotate them regularly

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is part of the HEAL-O healthcare initiative.

## Contact

For questions or support, please reach out through the repository issues or contact the maintainer.

---

**Note**: This is a backend service and requires proper configuration of external services (IBM Watson, Google Gemini, Twilio, Firebase) to function correctly. Ensure all API keys and credentials are properly set up before deployment.
