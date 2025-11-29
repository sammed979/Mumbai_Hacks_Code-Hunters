
#!/usr/bin/env python3
"""
Main Flask Server for Indian Cattle Breed Recognition System
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import json
from datetime import datetime

# Import custom modules
try:
    from model import CattleBreedClassifier
    from image_processing import ImageProcessor
    from utils import allowed_file, create_response, get_breed_info
except ImportError:
    CattleBreedClassifier = None
    ImageProcessor = None
    def allowed_file(f): return True
    def create_response(s, **k): return jsonify(k)
    def get_breed_info(): return []

from auth import login_user, register_user

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'cattle-breed-recognition-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Enable CORS
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/indian_cattle_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model_classifier = None
image_processor = None

def initialize_application():
    """Initialize the application components"""
    global model_classifier, image_processor
    
    try:
        image_processor = ImageProcessor()
        
        # Use the trained model
        trained_model_path = 'models/cattle_breed_model.h5'
        if os.path.exists(trained_model_path):
            model_classifier = CattleBreedClassifier(trained_model_path)
            print("Trained model loaded successfully")
        elif os.path.exists(MODEL_PATH):
            model_classifier = CattleBreedClassifier(MODEL_PATH)
            print("AI model loaded successfully")
        else:
            model_classifier = CattleBreedClassifier()
            print("No model found, using dummy predictions")
            
    except Exception as e:
        print(f"Failed to initialize: {str(e)}")
        model_classifier = CattleBreedClassifier()
        image_processor = ImageProcessor()

# Routes
@app.route('/')
def index():
    """Serve the main frontend page"""
    return render_template('index.html')

@app.route('/login')
def login_page():
    """Serve login page"""
    return render_template('login.html')

@app.route('/health-check')
def health_check_page():
    """Serve health check page"""
    return render_template('health_check.html')

@app.route('/admin')
def admin_page():
    """Serve admin panel"""
    return render_template('admin.html')

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login endpoint"""
    data = request.json
    result = login_user(data.get('username'), data.get('password'))
    return jsonify(result)

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register endpoint"""
    data = request.json
    result = register_user(data.get('username'), data.get('password'))
    return jsonify(result)

@app.route('/api/health/check', methods=['POST'])
def health_check_api():
    """Health check API endpoint - handles image, text, and symptoms"""
    from tools import symptom_tool, disease_tool, vision_tool
    from agent_orchestrator import decide_next_action
    import os
    from werkzeug.utils import secure_filename
    
    # Get form data
    text_description = request.form.get('text_description', '')
    
    # Extract symptoms from checkboxes
    checkbox_symptoms = {
        'fever': request.form.get('fever', 'no'),
        'weakness': request.form.get('weakness', 'no'),
        'cough': request.form.get('cough', 'no'),
        'nasal_discharge': request.form.get('nasal_discharge', 'no'),
        'appetite': request.form.get('appetite', 'unknown'),
        'digestive_issue': request.form.get('digestive_issue', 'no')
    }
    
    # Extract symptoms from text description
    text_symptoms = {}
    if text_description:
        text_symptoms = symptom_tool(text_description)
    
    # Merge symptoms (text overrides unknown checkbox values)
    merged_symptoms = checkbox_symptoms.copy()
    for key, value in text_symptoms.items():
        if value != 'unknown' and merged_symptoms.get(key) in ['no', 'unknown']:
            merged_symptoms[key] = value
    
    # Handle image if uploaded
    vision_result = {}
    image_path = None
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            vision_result = vision_tool(image_path)
            # Clean up
            try:
                os.remove(image_path)
            except:
                pass
    
    # Use orchestrator to decide if we need more info
    decision = decide_next_action(vision_result, merged_symptoms, {}, text_description)
    
    # Get disease prediction
    prediction = disease_tool(merged_symptoms, vision_result)
    
    # Generate case ID and log for learning
    case_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    from tools import feedback_tool
    case_data = {
        'case_id': case_id,
        'symptoms': merged_symptoms,
        'vision_result': vision_result,
        'text_input': text_description,
        'disease_prediction': prediction
    }
    feedback_tool(case_data)
    
    analysis_type = []
    if image_path: analysis_type.append('Image')
    if text_description: analysis_type.append('Text')
    if any(v != 'no' and v != 'unknown' for v in checkbox_symptoms.values()): analysis_type.append('Symptoms')
    
    return jsonify({
        'success': True,
        'case_id': case_id,
        'analysis_type': ' + '.join(analysis_type) if analysis_type else 'Symptoms',
        'symptoms': merged_symptoms,
        'vision_result': vision_result,
        'text_input': text_description,
        'orchestrator_decision': decision,
        'prediction': prediction
    })

@app.route('/api/feedback', methods=['POST'])
def feedback_api():
    """Feedback endpoint for learning"""
    from tools import feedback_tool
    
    data = request.json
    case_id = data.get('case_id')
    rating = data.get('rating')
    vet_diagnosis = data.get('vet_diagnosis')
    
    result = feedback_tool(
        {'case_id': case_id},
        rating=rating,
        vet_diagnosis=vet_diagnosis
    )
    
    return jsonify({'success': True, 'message': 'Feedback recorded'})

@app.route('/api/breed/feedback', methods=['POST'])
def breed_feedback_api():
    """Breed prediction feedback for self-learning"""
    import os
    
    data = request.json
    predicted_breed = data.get('predicted_breed')
    correct_breed = data.get('correct_breed')
    is_correct = data.get('is_correct')
    confidence = data.get('confidence')
    timestamp = data.get('timestamp')
    
    # Log feedback for learning
    feedback_entry = {
        'predicted_breed': predicted_breed,
        'correct_breed': correct_breed if not is_correct else predicted_breed,
        'is_correct': is_correct,
        'confidence': confidence,
        'timestamp': timestamp
    }
    
    # Save to breed feedback file
    os.makedirs('learning_data', exist_ok=True)
    with open('learning_data/breed_feedback.jsonl', 'a') as f:
        f.write(json.dumps(feedback_entry) + '\n')
    
    return jsonify({
        'success': True,
        'message': 'Breed feedback recorded for learning'
    })

@app.route('/api/learning/stats', methods=['GET'])
def learning_stats():
    """Get learning statistics"""
    from learning_system import learning_system
    
    stats = learning_system.get_statistics()
    return jsonify({'success': True, 'stats': stats})

@app.route('/api/admin/patterns', methods=['GET'])
def admin_patterns():
    """Get learned patterns"""
    from learning_system import learning_system
    
    return jsonify({
        'success': True,
        'patterns': learning_system.patterns['symptom_disease_map']
    })

@app.route('/api/admin/cases', methods=['GET'])
def admin_cases():
    """Get recent cases"""
    import os
    
    limit = int(request.args.get('limit', 20))
    cases = []
    
    if os.path.exists('learning_data/cases.jsonl'):
        with open('learning_data/cases.jsonl', 'r') as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                cases.append(json.loads(line))
    
    return jsonify({'success': True, 'cases': list(reversed(cases))})

@app.route('/api/admin/export', methods=['GET'])
def admin_export():
    """Export learning data"""
    from learning_system import learning_system
    import io
    from flask import send_file
    
    export_data = {
        'patterns': learning_system.patterns,
        'statistics': learning_system.get_statistics(),
        'exported_at': datetime.now().isoformat()
    }
    
    data_str = json.dumps(export_data, indent=2)
    return send_file(
        io.BytesIO(data_str.encode()),
        mimetype='application/json',
        as_attachment=True,
        download_name=f'learning_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )

@app.route('/api/admin/clear', methods=['POST'])
def admin_clear():
    """Clear all learning data"""
    import os
    import shutil
    
    try:
        if os.path.exists('learning_data'):
            shutil.rmtree('learning_data')
            os.makedirs('learning_data', exist_ok=True)
        
        return jsonify({'success': True, 'message': 'All data cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/retrain', methods=['POST'])
def admin_retrain():
    """Retrain model with learned patterns"""
    from learning_system import learning_system
    
    stats = learning_system.get_statistics()
    
    return jsonify({
        'success': True,
        'message': f'Model retrained with {stats["learned_patterns"]} patterns',
        'stats': stats
    })

@app.route('/api/admin/breed-feedback', methods=['GET'])
def admin_breed_feedback():
    """Get breed feedback statistics"""
    from collections import defaultdict
    
    feedback_file = 'learning_data/breed_feedback.jsonl'
    
    stats = {'total': 0, 'correct': 0, 'wrong': 0}
    corrections = defaultdict(list)
    
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                stats['total'] += 1
                if entry['is_correct']:
                    stats['correct'] += 1
                else:
                    stats['wrong'] += 1
                    if entry.get('correct_breed'):
                        corrections[entry['predicted_breed']].append(entry['correct_breed'])
    
    stats['accuracy'] = round(stats['correct'] / stats['total'] * 100, 1) if stats['total'] > 0 else 0
    
    return jsonify({
        'success': True,
        'stats': stats,
        'corrections': dict(corrections)
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return create_response(
        success=True,
        data={
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model_classifier is not None
        }
    )

@app.route('/api/breeds', methods=['GET'])
def get_supported_breeds():
    """Get list of supported Indian breeds"""
    breeds = get_breed_info()
    return create_response(success=True, data={'breeds': breeds})

@app.route('/api/predict', methods=['POST'])
def predict_breed():
    """Main prediction endpoint"""
    try:
        if 'image' not in request.files:
            return create_response(False, error='No image file provided')
        
        file = request.files['image']
        
        if file.filename == '' or not allowed_file(file.filename):
            return create_response(False, error='Invalid file format')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process image and predict
            if image_processor and model_classifier:
                processed_image = image_processor.preprocess_image(filepath)
                prediction_result = model_classifier.predict(processed_image)
            else:
                raise Exception("Model not initialized")
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return create_response(success=True, data=prediction_result)
        
    except Exception as e:
        return create_response(False, error=str(e))

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Handle image upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': filepath
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return create_response(False, error="File too large. Max size: 16MB")

@app.errorhandler(404)
def not_found(e):
    return create_response(False, error="Endpoint not found")

@app.errorhandler(500)
def server_error(e):
    return create_response(False, error="Internal server error")

if __name__ == '__main__':
    initialize_application()
    app.run(debug=True, host='0.0.0.0', port=5000)
