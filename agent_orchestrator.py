#!/usr/bin/env python3
"""
Orchestrator - Pure Decision Layer (Planner Only)
Minimal Data Integration: Uses only 6 key symptoms for decision
Disease model can use full rich data once predict is triggered
"""

def decide_next_action(vision_result, symptoms, case_meta, conversation_summary=""):
    """
    Pure orchestrator: decide ask_more or predict
    Returns JSON only
    """
    
    # Extract ONLY 6 key symptoms (Minimal Data Integration)
    fever = symptoms.get("fever", "unknown")
    appetite = symptoms.get("appetite", "unknown")
    cough = symptoms.get("cough", "unknown")
    nasal_discharge = symptoms.get("nasal_discharge", "unknown")
    weakness = symptoms.get("weakness", "unknown")
    digestive_issue = symptoms.get("digestive_issue", "unknown")
    
    # Count meaningful symptoms
    known = []
    missing = []
    
    if fever in ["yes", "no"]: known.append("fever")
    else: missing.append("fever")
    
    if appetite in ["normal", "low", "stopped"]: known.append("appetite")
    else: missing.append("appetite")
    
    if cough in ["yes", "no"]: known.append("cough")
    else: missing.append("cough")
    
    if nasal_discharge in ["yes", "no"]: known.append("nasal_discharge")
    else: missing.append("nasal_discharge")
    
    if weakness in ["yes", "no"]: known.append("weakness")
    else: missing.append("weakness")
    
    if digestive_issue in ["yes", "no"]: known.append("digestive_issue")
    else: missing.append("digestive_issue")
    
    # Decision: ask_more or predict
    if len(known) < 2:
        # Need more info
        q = []
        if "fever" in missing[:3]: q.append("Does the animal have fever?")
        if "appetite" in missing[:3] and len(q) < 3: q.append("Is it eating normally?")
        if "cough" in missing[:3] and len(q) < 3: q.append("Is the animal coughing?")
        
        return {
            "action": "ask_more",
            "missing_fields": missing[:3],
            "follow_up_questions": q,
            "confidence_note": f"Only {len(known)} symptoms known. Need at least 2."
        }
    
    # Check patterns
    if fever == "yes" and (cough == "yes" or nasal_discharge == "yes"):
        return {
            "action": "predict",
            "missing_fields": [],
            "follow_up_questions": [],
            "confidence_note": "Fever + respiratory symptoms. Proceed to disease model."
        }
    
    if fever == "yes" and digestive_issue == "yes":
        return {
            "action": "predict",
            "missing_fields": [],
            "follow_up_questions": [],
            "confidence_note": "Fever + digestive issue. Proceed to disease model."
        }
    
    if appetite in ["low", "stopped"] and (weakness == "yes" or digestive_issue == "yes"):
        return {
            "action": "predict",
            "missing_fields": [],
            "follow_up_questions": [],
            "confidence_note": "Appetite + weakness/digestive pattern. Proceed to disease model."
        }
    
    if len(known) >= 3:
        return {
            "action": "predict",
            "missing_fields": [],
            "follow_up_questions": [],
            "confidence_note": f"{len(known)} symptoms present. Sufficient for prediction."
        }
    
    # 2 symptoms but unclear - ask one more
    if len(known) == 2:
        q = []
        if "fever" in missing: q.append("Does the animal have fever?")
        elif "weakness" in missing: q.append("Is the animal weak or lying down?")
        
        if q:
            return {
                "action": "ask_more",
                "missing_fields": [missing[0]] if missing else [],
                "follow_up_questions": q,
                "confidence_note": "2 symptoms present but pattern unclear. Need one more."
            }
    
    # Default: proceed
    return {
        "action": "predict",
        "missing_fields": [],
        "follow_up_questions": [],
        "confidence_note": "Proceeding with available data."
    }
