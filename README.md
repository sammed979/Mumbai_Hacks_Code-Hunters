# 🐄 AI-Based Cattle Breed and Disease Recognizer

An intelligent web application that uses deep learning to identify cattle breeds and diagnose diseases through image analysis and symptom assessment. Built with Flask, TensorFlow, and modern web technologies.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Model Information](#model-information)
- [Self-Learning System](#self-learning-system)
- [Admin Panel](#admin-panel)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

This application provides farmers, veterinarians, and livestock managers with an AI-powered tool to:
- **Identify cattle breeds** from images with high accuracy
- **Diagnose diseases** using multi-modal analysis (symptoms + images)
- **Learn continuously** from user feedback to improve predictions
- **Track analytics** through an admin dashboard

The system currently supports **5 Indian cattle and buffalo breeds**:
- Gir (Cattle)
- Sahiwal (Cattle)
- Red Sindhi (Cattle)
- Tharparkar (Cattle)
- Murrah (Buffalo)

---

## ✨ Key Features

### 1. **Breed Recognition**
- Upload cattle/buffalo images for instant breed identification
- Get top 5 predictions with confidence scores
- View detailed breed information (origin, milk yield, characteristics)
- Provide feedback to improve model accuracy
- **Current Accuracy**: 97.65% on training data

### 2. **Disease Diagnosis**
- **Multi-modal analysis**: Combines symptoms, text descriptions, and images
- **Intelligent orchestrator**: Decides when to ask for more information vs. predict
- **Symptom-based detection**: Select from 20+ common symptoms
- **Text analysis**: Describe symptoms in natural language
- **Image analysis**: Upload photos of affected cattle
- **Confidence scoring**: Get reliability scores for predictions

### 3. **Self-Learning System**
- Learns from user feedback on breed predictions
- Tracks symptom-disease patterns over time
- Improves accuracy with each interaction
- Stores cases for future reference

### 4. **User Authentication**
- Secure login/registration system
- Password hashing with SHA256
- Session management
- User-specific data tracking

### 5. **Admin Dashboard**
- View system statistics and analytics
- Monitor learned patterns and accuracy
- Review recent cases and feedback
- Analyze breed prediction performance
- Export data for further analysis

---

## 🏗️ System Architecture

```
┌─────────────────┐
│   Web Browser   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Flask Server   │
│    (app.py)     │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────────┐
    ▼         ▼          ▼              ▼
┌────────┐ ┌──────┐ ┌─────────┐ ┌──────────────┐
│ Breed  │ │Health│ │Learning │ │     Auth     │
│ Model  │ │Tools │ │ System  │ │   System     │
└────────┘ └──────┘ └─────────┘ └──────────────┘
    │         │          │              │
    ▼         ▼          ▼              ▼
┌────────────────────────────────────────────┐
│         Data Storage Layer                 │
│  (Models, Learning Data, User Data)        │
└────────────────────────────────────────────┘
```

### Components:

1. **Frontend**: HTML/CSS/JavaScript with responsive design
2. **Backend**: Flask REST API
3. **ML Model**: TensorFlow/Keras CNN for breed classification
4. **Health System**: Rule-based + learning-based disease diagnosis
5. **Orchestrator**: Decision engine for multi-modal analysis
6. **Learning System**: Feedback collection and pattern recognition
7. **Authentication**: Secure user management

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum (8GB recommended)
- Modern web browser (Chrome, Firefox, Edge)

### Step 1: Clone or Download

```bash
cd path/to/project
cd Ai-based-cattle-breed-and-disease-recognizer-main
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- Flask (Web framework)
- TensorFlow (Deep learning)
- OpenCV (Image processing)
- Pillow (Image handling)
- NumPy (Numerical operations)
- scikit-learn (ML utilities)
- Flask-CORS (Cross-origin support)

### Step 3: Verify Installation

Check if all files are present:
```
✓ app.py
✓ model.py
✓ image_processing.py
✓ tools.py
✓ agent_orchestrator.py
✓ learning_system.py
✓ auth.py
✓ utils.py
✓ models/cattle_breed_model.h5
✓ models/breed_mapping.json
✓ templates/ (HTML files)
```

### Step 4: Run the Application

```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Model loaded successfully
```

### Step 5: Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

---

## 📖 Usage Guide

### Getting Started

#### 1. **Create an Account**
- Click "Register" on the login page
- Enter username and password
- Click "Register" to create account
- Login with your credentials

#### 2. **Breed Detection**

**Step-by-step:**
1. Navigate to the "Breed Detection" tab
2. Click "Choose File" or drag-and-drop an image
3. Supported formats: JPG, JPEG, PNG
4. Click "Detect Breed"
5. View results:
   - Primary breed with confidence score
   - Top 5 alternative predictions
   - Breed information (origin, milk yield, characteristics)

**Providing Feedback:**
- If prediction is correct: Click "Yes, Correct"
- If prediction is wrong:
  - Click "No, Wrong"
  - Enter the correct breed name
  - Click "Submit Feedback"
- Your feedback helps improve the model!

**Tips for Best Results:**
- Use clear, well-lit images
- Capture the full body of the animal
- Avoid blurry or distant shots
- Single animal per image works best

#### 3. **Health Check**

The health check system uses a **multi-modal approach** combining three types of input:

**A. Symptom Selection**
1. Navigate to "Health Check" tab
2. Select observed symptoms from checkboxes:
   - Fever
   - Loss of appetite
   - Reduced milk production
   - Coughing
   - Nasal discharge
   - Diarrhea
   - Constipation
   - Bloating
   - Lameness
   - Swelling
   - Skin lesions
   - Hair loss
   - Weight loss
   - Lethargy
   - Excessive salivation
   - Difficulty breathing
   - Eye discharge
   - Udder inflammation
   - Abnormal behavior
   - Tremors

**B. Text Description**
- Describe symptoms in your own words
- Example: "The cow has been coughing for 3 days and has watery eyes"
- The system extracts additional symptoms from text

**C. Image Upload**
- Upload photos showing visible symptoms
- Multiple images can be uploaded
- System analyzes visual indicators

**D. Get Diagnosis**
1. Click "Check Health"
2. System may respond with:
   - **"Need more information"**: Provide additional details
   - **Disease prediction**: Shows likely diseases with confidence scores

**E. Provide Feedback**
- After treatment, return to provide feedback
- Select "Helpful" or "Not Helpful"
- Add comments about actual diagnosis
- Helps system learn and improve

**Supported Diseases:**
- Foot and Mouth Disease (FMD)
- Mastitis
- Pneumonia
- Bloat
- Tuberculosis
- Brucellosis
- Anthrax
- Black Quarter
- Hemorrhagic Septicemia
- Parasitic infections
- And more...

#### 4. **Admin Dashboard**

**Access:** Login with admin credentials

**Features:**

**A. System Statistics**
- Total cases processed
- Feedback received
- Learned patterns count
- System accuracy metrics

**B. Learned Patterns**
- View symptom-disease associations
- See pattern confidence scores
- Track learning progress

**C. Recent Cases**
- Review recent health checks
- See symptoms and predictions
- Monitor system performance

**D. Breed Feedback Analysis**
- Correct vs incorrect predictions
- Most confused breeds
- Accuracy by breed
- User feedback trends

**E. Data Export**
- Download learning data
- Export cases for analysis
- Generate reports

---

## 🔌 API Documentation

### Authentication Endpoints

#### Register User
```http
POST /register
Content-Type: application/json

{
  "username": "farmer123",
  "password": "securepass"
}

Response:
{
  "success": true,
  "message": "User registered successfully"
}
```

#### Login
```http
POST /login
Content-Type: application/json

{
  "username": "farmer123",
  "password": "securepass"
}

Response:
{
  "success": true,
  "message": "Login successful"
}
```

### Breed Detection Endpoints

#### Predict Breed
```http
POST /predict
Content-Type: multipart/form-data

file: <image_file>

Response:
{
  "success": true,
  "primary_breed": "Gir",
  "confidence": 95.5,
  "alternatives": [
    {"breed": "Sahiwal", "confidence": 3.2, "rank": 2},
    {"breed": "Red_Sindhi", "confidence": 1.1, "rank": 3}
  ],
  "breed_info": {
    "type": "Zebu Cattle",
    "origin": "Gujarat, India",
    "milk_yield": "1200-1800 L/year",
    "characteristics": ["Curved horns", "Red/white patches"]
  }
}
```

#### Submit Breed Feedback
```http
POST /breed-feedback
Content-Type: application/json

{
  "predicted_breed": "Gir",
  "is_correct": false,
  "correct_breed": "Sahiwal",
  "confidence": 95.5,
  "image_path": "uploads/image.jpg"
}

Response:
{
  "success": true,
  "message": "Feedback recorded"
}
```

### Health Check Endpoints

#### Check Health
```http
POST /health-check
Content-Type: multipart/form-data

symptoms[]: fever
symptoms[]: coughing
description: "Cow has fever and cough for 3 days"
image: <optional_image_file>

Response:
{
  "action": "predict",
  "diseases": [
    {
      "disease": "Pneumonia",
      "confidence": 85.5,
      "symptoms_matched": ["fever", "coughing", "difficulty_breathing"]
    }
  ],
  "message": "Based on symptoms, likely diagnosis is Pneumonia"
}
```

#### Submit Health Feedback
```http
POST /health-feedback
Content-Type: application/json

{
  "case_id": "case_123",
  "helpful": true,
  "actual_diagnosis": "Pneumonia",
  "comments": "Treatment was successful"
}

Response:
{
  "success": true,
  "message": "Feedback recorded"
}
```

### Admin Endpoints

#### Get Statistics
```http
GET /admin/stats

Response:
{
  "total_cases": 150,
  "total_feedback": 120,
  "learned_patterns": 45,
  "accuracy": 87.5,
  "breed_accuracy": 97.65
}
```

#### Get Learning Data
```http
GET /admin/learning-data

Response:
{
  "patterns": [...],
  "recent_cases": [...],
  "breed_feedback": [...]
}
```

---

## 🧠 Model Information

### Breed Classification Model

**Architecture:**
- Input: 224x224x3 RGB images
- Base: Custom CNN with BatchNormalization
- Layers:
  - Conv2D (16 filters, 5x5) + BatchNorm + MaxPool
  - Conv2D (32 filters, 3x3) + BatchNorm + MaxPool
  - Dense (128 units) + Dropout (0.5)
  - Dense (64 units)
  - Output (5 classes, softmax)

**Training Details:**
- Dataset: 85 images (5 breeds)
- Epochs: 100
- Batch size: 2
- Optimizer: Adam (lr=0.001)
- Loss: Categorical crossentropy
- Data augmentation: Rotation, shifts, flips
- Validation split: 15%

**Performance:**
- Training accuracy: 97.65%
- Model size: 9.87 MB
- Inference time: ~100ms per image

**Supported Breeds:**
1. **Gir** - Zebu cattle from Gujarat
2. **Sahiwal** - Dairy cattle from Punjab
3. **Red Sindhi** - Dual-purpose cattle from Sindh
4. **Tharparkar** - Drought-resistant cattle from Rajasthan
5. **Murrah** - High-yielding buffalo from Haryana

### Disease Diagnosis System

**Approach:** Hybrid (Rule-based + Learning-based)

**Components:**

1. **Symptom Tool**: Extracts symptoms from text using NLP
2. **Vision Tool**: Analyzes images for visual symptoms
3. **Disease Tool**: Matches symptoms to diseases using:
   - Predefined disease-symptom mappings
   - Learned patterns from user feedback
4. **Orchestrator**: Decides when to predict vs. ask for more info

**Decision Logic:**
- Minimum 3 symptoms required for prediction
- Confidence threshold: 60%
- Uses 6 key symptoms for orchestrator decisions
- Full symptom set for disease prediction

**Learning Mechanism:**
- Tracks symptom-disease co-occurrences
- Updates pattern confidence based on feedback
- Calculates accuracy from user confirmations
- Stores cases for future reference

---

## 📊 Self-Learning System

### How It Works

The system continuously improves through user feedback:

#### 1. **Breed Learning**
- Records every prediction and user feedback
- Tracks correct vs incorrect predictions
- Identifies commonly confused breeds
- Stores data in `learning_data/breed_feedback.jsonl`

**Data Format:**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "predicted_breed": "Gir",
  "is_correct": false,
  "correct_breed": "Sahiwal",
  "confidence": 95.5,
  "image_path": "uploads/img.jpg"
}
```

#### 2. **Disease Learning**
- Logs all health check cases
- Records symptom combinations
- Tracks prediction accuracy
- Updates disease-symptom patterns

**Pattern Storage:**
```json
{
  "Pneumonia": {
    "fever": 0.95,
    "coughing": 0.90,
    "difficulty_breathing": 0.85
  }
}
```

#### 3. **Feedback Loop**
```
User Input → Prediction → User Feedback → Pattern Update → Improved Predictions
```

### Accessing Learning Data

**Admin Dashboard:**
- View learned patterns
- See accuracy trends
- Review feedback statistics

**File System:**
- `learning_data/breed_feedback.jsonl` - Breed feedback
- `learning_data/cases.jsonl` - Health cases
- `learning_data/patterns.json` - Disease patterns
- `learning_data/feedback.jsonl` - Health feedback

---

## 👨‍💼 Admin Panel

### Access

1. Login with admin credentials
2. Navigate to `/admin` route
3. Or click "Admin Panel" in navigation

### Features

#### Dashboard Overview
- **Total Cases**: Number of health checks performed
- **Feedback Count**: User feedback received
- **Learned Patterns**: Disease-symptom associations
- **Accuracy Metrics**: System performance

#### Learned Patterns Section
- View all disease-symptom patterns
- See confidence scores for each association
- Track pattern evolution over time

#### Recent Cases
- Last 10 health check cases
- Symptoms, predictions, and outcomes
- Case IDs for reference

#### Breed Feedback Analysis
- Correct prediction rate
- Most confused breed pairs
- Accuracy by breed
- Feedback trends over time

#### Data Management
- Export learning data
- Clear old cases
- Reset patterns (use with caution)
- Backup system data

---

## 📁 Project Structure

```
Ai-based-cattle-breed-and-disease-recognizer-main/
│
├── app.py                      # Main Flask application
├── model.py                    # Breed classification model
├── image_processing.py         # Image preprocessing
├── tools.py                    # Health check tools
├── agent_orchestrator.py       # Decision engine
├── learning_system.py          # Self-learning system
├── auth.py                     # Authentication
├── utils.py                    # Utility functions
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── users.json                  # User database
│
├── models/                     # Trained models
│   ├── cattle_breed_model.h5   # CNN model (9.87 MB)
│   └── breed_mapping.json      # Class labels
│
├── templates/                  # HTML templates
│   ├── index.html              # Main page
│   ├── login.html              # Login/register
│   ├── admin.html              # Admin dashboard
│   └── health_check.html       # Health check page
│
├── learning_data/              # Learning system data
│   ├── breed_feedback.jsonl    # Breed feedback
│   ├── cases.jsonl             # Health cases
│   ├── patterns.json           # Disease patterns
│   └── feedback.jsonl          # Health feedback
│
├── uploads/                    # Uploaded images
│
└── data/                       # Data directory
```

---

## 🛠️ Technologies Used

### Backend
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning
- **OpenCV** - Image processing
- **Pillow** - Image handling
- **NumPy** - Numerical operations
- **scikit-learn** - ML utilities

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling
- **JavaScript** - Interactivity
- **Fetch API** - AJAX requests

### Data Storage
- **JSON** - User data and learning data
- **JSONL** - Log files
- **HDF5** - Model storage

### Security
- **SHA256** - Password hashing
- **Flask sessions** - Session management
- **CORS** - Cross-origin security

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Not Loading
**Error:** `Model file not found`
**Solution:**
- Verify `models/cattle_breed_model.h5` exists
- Check file size (should be ~9.87 MB)
- Re-download model if corrupted

#### 2. Import Errors
**Error:** `ModuleNotFoundError: No module named 'tensorflow'`
**Solution:**
```bash
pip install -r requirements.txt
```

#### 3. Port Already in Use
**Error:** `Address already in use`
**Solution:**
```bash
# Change port in app.py
app.run(debug=True, port=5001)
```

#### 4. Image Upload Fails
**Error:** `File type not allowed`
**Solution:**
- Use JPG, JPEG, or PNG format
- Check file size (max 16MB)
- Ensure image is not corrupted

#### 5. Low Prediction Confidence
**Issue:** All predictions below 50%
**Solution:**
- Use clearer images
- Ensure good lighting
- Capture full body of animal
- Avoid multiple animals in frame

#### 6. Health Check Not Working
**Issue:** Always asks for more information
**Solution:**
- Provide at least 3 symptoms
- Add text description
- Upload relevant images
- Be specific in descriptions

### Performance Optimization

**For Faster Predictions:**
- Reduce image size before upload
- Use GPU if available (TensorFlow GPU)
- Close unnecessary applications
- Increase system RAM

**For Better Accuracy:**
- Provide user feedback regularly
- Use high-quality images
- Be detailed in symptom descriptions
- Review and correct predictions

---

## 🚀 Future Enhancements

### Planned Features

1. **Expanded Breed Support**
   - Add 69 more Indian breeds
   - Support international breeds
   - Regional breed variations

2. **Advanced Disease Diagnosis**
   - Integration with veterinary databases
   - Treatment recommendations
   - Medication suggestions
   - Vaccination schedules

3. **Mobile Application**
   - Android app
   - iOS app
   - Offline mode
   - Camera integration

4. **Multi-language Support**
   - Hindi
   - Regional languages
   - Voice input

5. **Analytics Dashboard**
   - Herd management
   - Health tracking over time
   - Milk production correlation
   - Cost analysis

6. **Community Features**
   - Farmer forums
   - Expert consultations
   - Case sharing
   - Best practices

7. **IoT Integration**
   - Wearable sensors
   - Automated monitoring
   - Real-time alerts
   - Temperature tracking

8. **Blockchain Integration**
   - Cattle registry
   - Ownership verification
   - Health records
   - Breeding history

---

## 📞 Support

For issues, questions, or contributions:
- Check troubleshooting section
- Review API documentation
- Contact system administrator

---

## 📄 License

This project is developed for educational and agricultural purposes.

---

## 🙏 Acknowledgments

- Indian Council of Agricultural Research (ICAR) for breed information
- Veterinary experts for disease knowledge
- Farmers for feedback and testing
- Open-source community for tools and libraries

---

## 📈 Version History

**v1.0.0** (Current)
- Initial release
- 5 breed support
- Multi-modal health check
- Self-learning system
- Admin dashboard
- 97.65% breed accuracy

---

**Made with ❤️ for Indian farmers and livestock managers**

*Empowering agriculture through AI*
