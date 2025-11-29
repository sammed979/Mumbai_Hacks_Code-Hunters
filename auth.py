"""
Minimal Authentication System
"""

import json
import hashlib
import os
from datetime import datetime, timedelta

USERS_FILE = "users.json"

def hash_password(password):
    """Hash password"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def register_user(username, password, role="farmer"):
    """Register new user"""
    users = load_users()
    
    if username in users:
        return {"success": False, "error": "User already exists"}
    
    users[username] = {
        "password": hash_password(password),
        "role": role,
        "created_at": datetime.now().isoformat()
    }
    
    save_users(users)
    return {"success": True, "message": "User registered"}

def login_user(username, password):
    """Login user"""
    users = load_users()
    
    if username not in users:
        return {"success": False, "error": "Invalid credentials"}
    
    if users[username]["password"] != hash_password(password):
        return {"success": False, "error": "Invalid credentials"}
    
    return {
        "success": True,
        "user": {
            "username": username,
            "role": users[username]["role"]
        }
    }

# Create default admin user
if not os.path.exists(USERS_FILE):
    register_user("admin", "admin123", "admin")
    register_user("farmer", "farmer123", "farmer")
