#!/usr/bin/env python3
"""
Simulate a deteriorating patient trajectory for sparkline testing.

This script submits multiple triage assessments for a single patient
to generate trend data that will display as sparklines in the dashboard.

Usage:
    python scripts/simulate_patient_trajectory.py

The script will:
1. Login as a test patient to get a JWT
2. Submit 5 sequential triage assessments showing deterioration
3. Print the patient_id for verification in the Staff Dashboard
"""

import time
import requests
import sys

API_BASE = "http://localhost:8000"

# Simulated deteriorating vitals over 4 hours (Shock progression)
TRAJECTORY = [
    {
        "label": "T=0h (Normal)",
        "vitals": {"Pulse": 80, "SBP": 120, "DBP": 80, "Resp": 16, "O2Sat": 98, "Temp": 98.6, "PainScale": 3},
        "complaint": "Mild abdominal discomfort, feeling slightly unwell"
    },
    {
        "label": "T=1h (Early Warning)",
        "vitals": {"Pulse": 95, "SBP": 115, "DBP": 75, "Resp": 18, "O2Sat": 97, "Temp": 99.1, "PainScale": 5},
        "complaint": "Increasing abdominal pain, slight fever developing"
    },
    {
        "label": "T=2h (Warning)",
        "vitals": {"Pulse": 110, "SBP": 105, "DBP": 70, "Resp": 22, "O2Sat": 95, "Temp": 100.2, "PainScale": 7},
        "complaint": "Severe abdominal pain, tachycardia noted, patient anxious"
    },
    {
        "label": "T=3h (Critical)",
        "vitals": {"Pulse": 125, "SBP": 95, "DBP": 60, "Resp": 26, "O2Sat": 93, "Temp": 101.5, "PainScale": 9},
        "complaint": "Rigid abdomen, hypotension developing, altered mental status"
    },
    {
        "label": "T=4h (Shock)",
        "vitals": {"Pulse": 140, "SBP": 85, "DBP": 50, "Resp": 30, "O2Sat": 90, "Temp": 102.8, "PainScale": 10},
        "complaint": "Septic shock, requires immediate resuscitation, unresponsive to verbal"
    },
]


def create_test_user(email: str, password: str) -> bool:
    """Create a test user if they don't exist."""
    res = requests.post(f"{API_BASE}/auth/register", json={
        "email": email,
        "password": password,
        "role": "PATIENT"
    })
    if res.status_code in (200, 201):
        print(f"✓ Created new user: {email}")
        return True
    elif res.status_code == 400 and "already registered" in res.text.lower():
        print(f"✓ User already exists: {email}")
        return True
    else:
        print(f"✗ Failed to create user: {res.status_code} - {res.text}")
        return False


def login(email: str, password: str) -> str | None:
    """Login and return JWT token."""
    res = requests.post(f"{API_BASE}/auth/token", data={
        "username": email,
        "password": password
    })
    if res.status_code == 200:
        token = res.json().get("access_token")
        print(f"✓ Logged in successfully")
        return token
    else:
        print(f"✗ Login failed: {res.status_code} - {res.text}")
        return None


def create_patient_profile(token: str, first_name: str, last_name: str) -> int | None:
    """Create or get patient profile, return patient_id."""
    headers = {"Authorization": f"Bearer {token}"}
    
    # Try to get existing profile first
    res = requests.get(f"{API_BASE}/auth/me", headers=headers)
    if res.status_code == 200:
        user_data = res.json()
        if user_data.get("patient_profile"):
            patient_id = user_data["patient_profile"]["id"]
            print(f"✓ Found existing patient profile (ID: {patient_id})")
            return patient_id
    
    # Create new profile
    res = requests.post(f"{API_BASE}/auth/patient-profile", headers=headers, json={
        "first_name": first_name,
        "last_name": last_name,
        "phone": "555-0123",
        "allergies": "None known"
    })
    if res.status_code in (200, 201):
        patient_id = res.json().get("id")
        print(f"✓ Created patient profile (ID: {patient_id})")
        return patient_id
    elif res.status_code == 400:
        # Profile might already exist, try to get it
        res = requests.get(f"{API_BASE}/auth/me", headers=headers)
        if res.status_code == 200:
            patient_id = res.json().get("patient_profile", {}).get("id")
            if patient_id:
                print(f"✓ Found existing patient profile (ID: {patient_id})")
                return patient_id
    
    print(f"✗ Failed to create profile: {res.status_code} - {res.text}")
    return None


def submit_triage(token: str, vitals: dict, complaint: str, age: int = 45) -> dict | None:
    """Submit a triage assessment."""
    headers = {"Authorization": f"Bearer {token}"}
    
    payload = {
        "chief_complaint": complaint,
        "static_vitals": {
            "Age": age,
            **vitals
        }
    }
    
    res = requests.post(f"{API_BASE}/triage/assess", headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()
    else:
        print(f"✗ Triage failed: {res.status_code} - {res.text}")
        return None


def main():
    print("=" * 60)
    print("🏥 Patient Trajectory Simulator")
    print("   Generating deteriorating vitals for sparkline testing")
    print("=" * 60)
    print()
    
    # Test credentials
    email = "trajectory.test@hospital.demo"
    password = "TestPatient123!"
    
    # Step 1: Create/verify user
    print("Step 1: Setting up test user...")
    if not create_test_user(email, password):
        sys.exit(1)
    
    # Step 2: Login
    print("\nStep 2: Authenticating...")
    token = login(email, password)
    if not token:
        sys.exit(1)
    
    # Step 3: Create patient profile
    print("\nStep 3: Setting up patient profile...")
    patient_id = create_patient_profile(token, "John", "Deteriorating")
    if not patient_id:
        sys.exit(1)
    
    # Step 4: Submit trajectory
    print("\nStep 4: Submitting deteriorating trajectory...")
    print("-" * 60)
    
    results = []
    for i, point in enumerate(TRAJECTORY):
        print(f"\n  📊 {point['label']}")
        print(f"     HR: {point['vitals']['Pulse']} | SBP: {point['vitals']['SBP']} | O2: {point['vitals']['O2Sat']}%")
        
        result = submit_triage(token, point['vitals'], point['complaint'])
        if result:
            results.append(result)
            print(f"     ✓ Triage Level: ESI-{result['triage_level']} | Token: {result['token_number']}")
        else:
            print(f"     ✗ Failed to submit")
        
        # Small delay between submissions
        if i < len(TRAJECTORY) - 1:
            time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAJECTORY SIMULATION COMPLETE")
    print("=" * 60)
    print(f"""
    Patient ID: {patient_id}
    Email: {email}
    Assessments: {len(results)} submitted
    
    📈 Expected Sparkline Behavior:
       • Heart Rate (Red):   80 → 140 (RISING - Bad)
       • Blood Pressure (Blue): 120 → 85 (FALLING - Bad)  
       • O2 Saturation (Green): 98 → 90 (FALLING - Bad)
    
    🔍 To verify:
       1. Open Staff Dashboard: http://localhost:3000
       2. Login as staff (or use dev mode)
       3. Find patient "John Deteriorating" or token {results[-1]['token_number'] if results else 'N/A'}
       4. Click to open modal → See "Vitals Trends" sparklines
       
    ⚠️  The Shock Index crossover (HR/SBP) should be clearly visible!
    """)


if __name__ == "__main__":
    main()
