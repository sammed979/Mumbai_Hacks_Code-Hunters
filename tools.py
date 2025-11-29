#!/usr/bin/env python3
"""
Agent Tools: Vision, Symptom, Disease, Feedback
"""

import json
import re
from datetime import datetime

# Vision Tool
def vision_tool(image_path):
    """Extract visual info from cattle/buffalo image"""
    # Placeholder - integrate with actual vision model
    return {
        "species": "cattle",
        "breed": "unknown",
        "disease_signs": [],
        "body_condition": "normal",
        "visible_issues": []
    }

# Symptom Tool
def symptom_tool(user_text):
    """Extract symptoms from farmer's text"""
    text = user_text.lower()
    
    symptoms = {
        "fever": "unknown",
        "appetite": "unknown",
        "cough": "unknown",
        "nasal_discharge": "unknown",
        "weakness": "unknown",
        "digestive_issue": "unknown"
    }
    
    # Fever detection
    if any(word in text for word in ["fever", "hot", "temperature", "गर्मी"]):
        symptoms["fever"] = "yes"
    elif "no fever" in text:
        symptoms["fever"] = "no"
    
    # Appetite detection
    if any(word in text for word in ["not eating", "stopped eating", "no appetite", "खाना नहीं"]):
        symptoms["appetite"] = "stopped"
    elif any(word in text for word in ["eating less", "low appetite"]):
        symptoms["appetite"] = "low"
    elif any(word in text for word in ["eating normal", "eating well"]):
        symptoms["appetite"] = "normal"
    
    # Cough detection
    if any(word in text for word in ["cough", "coughing", "खांसी"]):
        symptoms["cough"] = "yes"
    
    # Nasal discharge
    if any(word in text for word in ["nasal", "discharge", "runny nose", "mucus", "नाक"]):
        symptoms["nasal_discharge"] = "yes"
    
    # Weakness
    if any(word in text for word in ["weak", "lying down", "can't stand", "कमजोर"]):
        symptoms["weakness"] = "yes"
    
    # Digestive issues
    if any(word in text for word in ["diarrhea", "loose stool", "bloat", "constipation", "दस्त"]):
        symptoms["digestive_issue"] = "yes"
    
    return symptoms

# Disease Tool
def disease_tool(symptoms, vision_result):
    """Predict probable disease - uses FULL RICH DATA + LEARNED PATTERNS
    Self-learning: checks learned patterns first, falls back to rules
    """
    from learning_system import learning_system
    
    # Try learned prediction first
    learned = learning_system.get_learned_prediction(symptoms)
    if learned and learned["confidence"] > 0.6:
        # Use learned prediction if confidence is high
        return {
            "disease": learned["disease"],
            "risk_level": "Medium",  # Can be refined based on learning
            "reason": f"Based on {learned['pattern_matches']} similar cases",
            "care_steps": ["Follow previously successful care steps", "Monitor closely"],
            "vet_urgency": "Call vet if symptoms worsen",
            "learned": True,
            "confidence": learned["confidence"]
        }
    
    # Fall back to rule-based logic
    fever = symptoms.get("fever") == "yes"
    cough = symptoms.get("cough") == "yes"
    nasal = symptoms.get("nasal_discharge") == "yes"
    appetite_low = symptoms.get("appetite") in ["low", "stopped"]
    weakness = symptoms.get("weakness") == "yes"
    digestive = symptoms.get("digestive_issue") == "yes"
    
    # Can also use vision_result for richer analysis
    visible_issues = vision_result.get("visible_issues", [])
    disease_signs = vision_result.get("disease_signs", [])
    
    # Pattern matching with full data
    # Vision + symptoms combined
    if (fever and cough and nasal) or "respiratory_distress" in disease_signs:
        return {
            "disease": "Respiratory Infection (possible Pneumonia)",
            "risk_level": "High",
            "reason": "Fever with cough and nasal discharge indicates respiratory infection",
            "care_steps": [
                "Keep animal in dry, warm place",
                "Ensure clean water available",
                "Isolate from other animals"
            ],
            "vet_urgency": "Call vet immediately - needs antibiotics"
        }
    
    if fever and digestive:
        return {
            "disease": "Digestive Infection (possible Enteritis)",
            "risk_level": "High",
            "reason": "Fever with digestive issues suggests infection",
            "care_steps": [
                "Provide clean water frequently",
                "Stop solid feed temporarily",
                "Keep animal clean"
            ],
            "vet_urgency": "Call vet today - may need IV fluids"
        }
    
    if appetite_low and weakness:
        return {
            "disease": "General Weakness or Nutritional Deficiency",
            "risk_level": "Medium",
            "reason": "Loss of appetite with weakness",
            "care_steps": [
                "Offer fresh green fodder",
                "Check for mineral deficiency",
                "Monitor for other symptoms"
            ],
            "vet_urgency": "Call vet if no improvement in 24 hours"
        }
    
    if cough:
        return {
            "disease": "Mild Respiratory Issue",
            "risk_level": "Low",
            "reason": "Cough without other major symptoms",
            "care_steps": [
                "Keep in dust-free area",
                "Ensure good ventilation",
                "Monitor closely"
            ],
            "vet_urgency": "Call vet if cough worsens or fever develops"
        }
    
    # Default
    return {
        "disease": "Unable to determine specific issue",
        "risk_level": "Medium",
        "reason": "Symptoms present but pattern unclear",
        "care_steps": [
            "Monitor animal closely",
            "Note any new symptoms",
            "Keep animal comfortable"
        ],
        "vet_urgency": "Call vet for proper examination"
    }

# Feedback Tool
def feedback_tool(case_data, rating=None, vet_diagnosis=None):
    """Store case for learning - integrates with self-learning system"""
    from learning_system import learning_system
    
    case_data["feedback"] = {
        "rating": rating,
        "vet_diagnosis": vet_diagnosis,
        "labeled": vet_diagnosis is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Log to learning system
    learning_system.log_case(case_data)
    
    # If feedback provided, log separately
    if rating or vet_diagnosis:
        learning_system.log_feedback(
            case_data.get("case_id"),
            {
                "rating": rating,
                "actual_diagnosis": vet_diagnosis,
                "predicted_disease": case_data.get("disease_prediction", {}).get("disease")
            }
        )
    
    return {"status": "saved", "learned": True}
