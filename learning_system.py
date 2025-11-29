#!/usr/bin/env python3
"""
Self-Learning System - Learns from every interaction
"""

import json
import os
from datetime import datetime
from collections import defaultdict

CASES_FILE = "learning_data/cases.jsonl"
PATTERNS_FILE = "learning_data/patterns.json"
FEEDBACK_FILE = "learning_data/feedback.jsonl"

os.makedirs("learning_data", exist_ok=True)

class SelfLearningSystem:
    """Self-learning system that improves from every case"""
    
    def __init__(self):
        self.patterns = self.load_patterns()
    
    def load_patterns(self):
        """Load learned patterns"""
        if os.path.exists(PATTERNS_FILE):
            with open(PATTERNS_FILE, 'r') as f:
                return json.load(f)
        return {
            "symptom_disease_map": {},
            "accuracy_scores": {},
            "common_patterns": [],
            "question_effectiveness": {}
        }
    
    def save_patterns(self):
        """Save learned patterns"""
        with open(PATTERNS_FILE, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def log_case(self, case_data):
        """Log every case for learning"""
        case_data["timestamp"] = datetime.now().isoformat()
        
        with open(CASES_FILE, 'a') as f:
            f.write(json.dumps(case_data) + "\n")
        
        # Learn from this case
        self._learn_from_case(case_data)
    
    def log_feedback(self, case_id, feedback_data):
        """Log user feedback"""
        feedback = {
            "case_id": case_id,
            "timestamp": datetime.now().isoformat(),
            **feedback_data
        }
        
        with open(FEEDBACK_FILE, 'a') as f:
            f.write(json.dumps(feedback) + "\n")
        
        # Learn from feedback
        self._learn_from_feedback(feedback)
    
    def _learn_from_case(self, case):
        """Extract patterns from case"""
        symptoms = case.get("symptoms", {})
        prediction = case.get("disease_prediction", {})
        
        # Build symptom pattern key
        active_symptoms = [k for k, v in symptoms.items() if v in ["yes", "stopped", "low"]]
        pattern_key = "+".join(sorted(active_symptoms))
        
        if pattern_key and prediction.get("disease"):
            # Track symptom-disease correlation
            if pattern_key not in self.patterns["symptom_disease_map"]:
                self.patterns["symptom_disease_map"][pattern_key] = {}
            
            disease = prediction["disease"]
            if disease not in self.patterns["symptom_disease_map"][pattern_key]:
                self.patterns["symptom_disease_map"][pattern_key][disease] = 0
            
            self.patterns["symptom_disease_map"][pattern_key][disease] += 1
        
        # Track common patterns
        if len(active_symptoms) >= 2:
            if pattern_key not in [p["pattern"] for p in self.patterns["common_patterns"]]:
                self.patterns["common_patterns"].append({
                    "pattern": pattern_key,
                    "count": 1,
                    "diseases": [prediction.get("disease")]
                })
            else:
                for p in self.patterns["common_patterns"]:
                    if p["pattern"] == pattern_key:
                        p["count"] += 1
                        if prediction.get("disease") not in p["diseases"]:
                            p["diseases"].append(prediction.get("disease"))
        
        self.save_patterns()
    
    def _learn_from_feedback(self, feedback):
        """Learn from user feedback"""
        case_id = feedback.get("case_id")
        rating = feedback.get("rating")
        actual_diagnosis = feedback.get("actual_diagnosis")
        
        # Update accuracy scores
        if actual_diagnosis:
            if actual_diagnosis not in self.patterns["accuracy_scores"]:
                self.patterns["accuracy_scores"][actual_diagnosis] = {"correct": 0, "total": 0}
            
            self.patterns["accuracy_scores"][actual_diagnosis]["total"] += 1
            if rating and rating >= 4:
                self.patterns["accuracy_scores"][actual_diagnosis]["correct"] += 1
        
        self.save_patterns()
    
    def get_learned_prediction(self, symptoms):
        """Get prediction based on learned patterns"""
        active_symptoms = [k for k, v in symptoms.items() if v in ["yes", "stopped", "low"]]
        pattern_key = "+".join(sorted(active_symptoms))
        
        if pattern_key in self.patterns["symptom_disease_map"]:
            diseases = self.patterns["symptom_disease_map"][pattern_key]
            # Return most common disease for this pattern
            most_common = max(diseases.items(), key=lambda x: x[1])
            return {
                "disease": most_common[0],
                "confidence": most_common[1] / sum(diseases.values()),
                "learned": True,
                "pattern_matches": most_common[1]
            }
        
        return None
    
    def get_statistics(self):
        """Get learning statistics"""
        total_cases = 0
        if os.path.exists(CASES_FILE):
            with open(CASES_FILE, 'r') as f:
                total_cases = sum(1 for _ in f)
        
        total_feedback = 0
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r') as f:
                total_feedback = sum(1 for _ in f)
        
        return {
            "total_cases": total_cases,
            "total_feedback": total_feedback,
            "learned_patterns": len(self.patterns["symptom_disease_map"]),
            "common_patterns": len(self.patterns["common_patterns"]),
            "accuracy_data": self.patterns["accuracy_scores"]
        }

# Global instance
learning_system = SelfLearningSystem()
