import os
import json
import urllib.request
import warnings
from datetime import datetime
from xmlrpc import client
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import urllib.error
import requests
import subprocess, json
import os
from google import genai
from google.genai import types

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ---==- Datasets ---------
DATA_PATH = "dataset/hospital_deterioration_ml_ready.csv"
PATIENTS_PATH = "dataset/patients.csv"
VITALS_PATH = "dataset/vitals_timeseries.csv"
LABS_PATH = "dataset/labs_timeseries.csv"
MODEL_PATH = "model.joblib"
TARGET_COL = "deterioration_next_12h"

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '').strip()
GROQ_MODEL = "llama-3.1-8b-instant"

RISK_THRESHOLDS = {
    "CRITICAL": 0.70,
    "HIGH": 0.45,
    "MODERATE": 0.25,
    "LOW": 0.10,
}

UNITS = ['ICU', 'CCU', 'Telemetry', 'Med-Surg', 'ER', 'SICU', 'MICU', 'Step-Down']

df_ml_data = None
df_patients = None
df_vitals = None
df_labs = None
model = None

# Fast lookup indexes
vitals_index = {}  # patient_id -> row index of latest vitals
labs_index = {}    # patient_id -> row index of latest labs

# PRE-COMPUTED cache
all_predictions_cache = []
kpis_cache = {}


def get_risk_tier(probability):
    if probability >= RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif probability >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif probability >= RISK_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    elif probability >= RISK_THRESHOLDS["LOW"]:
        return "LOW"
    return "MINIMAL"

_TIER_ORDER = ["MINIMAL", "LOW", "MODERATE", "HIGH", "CRITICAL"]
_TIER_RANK = {t:i for i,t in enumerate(_TIER_ORDER)}

def _max_tier(a, b):
    return a if _TIER_RANK[a] >= _TIER_RANK[b] else b

def apply_clinical_override(vitals, labs, model_tier):
    tier = model_tier

    spo2 = vitals.get("spo2_pct")
    rr   = vitals.get("respiratory_rate")
    sbp  = vitals.get("systolic_bp")
    hr   = vitals.get("heart_rate")
    lac  = labs.get("lactate")

    # Red flags (rule-based safety)
    if spo2 is not None and spo2 < 75:
        tier = _max_tier(tier, "CRITICAL")
    elif spo2 is not None and spo2 < 85:
        tier = _max_tier(tier, "HIGH")

    if rr is not None and rr > 35:
        tier = _max_tier(tier, "CRITICAL")
    elif rr is not None and rr > 30:
        tier = _max_tier(tier, "HIGH")

    if sbp is not None and sbp < 90:
        tier = _max_tier(tier, "HIGH")

    if lac is not None and lac >= 4:
        tier = _max_tier(tier, "CRITICAL")

    # bradycardia with severe hypoxemia = worse
    if (hr is not None and hr < 50) and (spo2 is not None and spo2 < 85):
        tier = _max_tier(tier, "CRITICAL")

    return tier

def get_patient_latest_vitals(patient_id):
    if df_vitals is None:
        return {}
    if patient_id not in vitals_index:
        return {}
    idx = vitals_index[patient_id]
    latest = df_vitals.iloc[idx]
    return {
        'heart_rate': float(latest['heart_rate']) if pd.notna(latest['heart_rate']) else None,
        'respiratory_rate': float(latest['respiratory_rate']) if pd.notna(latest['respiratory_rate']) else None,
        'spo2_pct': float(latest['spo2_pct']) if pd.notna(latest['spo2_pct']) else None,
        'temperature_c': float(latest['temperature_c']) if pd.notna(latest['temperature_c']) else None,
        'systolic_bp': float(latest['systolic_bp']) if pd.notna(latest['systolic_bp']) else None,
        'diastolic_bp': float(latest['diastolic_bp']) if pd.notna(latest['diastolic_bp']) else None,
        'oxygen_flow': float(latest['oxygen_flow']) if pd.notna(latest['oxygen_flow']) else None,
    }


def get_patient_latest_labs(patient_id):
    if df_labs is None:
        return {}
    if patient_id not in labs_index:
        return {}
    idx = labs_index[patient_id]
    latest = df_labs.iloc[idx]
    return {
        'wbc_count': float(latest['wbc_count']) if pd.notna(latest['wbc_count']) else None,
        'lactate': float(latest['lactate']) if pd.notna(latest['lactate']) else None,
        'creatinine': float(latest['creatinine']) if pd.notna(latest['creatinine']) else None,
        'crp_level': float(latest['crp_level']) if pd.notna(latest['crp_level']) else None,
        'hemoglobin': float(latest['hemoglobin']) if pd.notna(latest['hemoglobin']) else None,
        'sepsis_risk_score': float(latest['sepsis_risk_score']) if pd.notna(latest['sepsis_risk_score']) else None,
    }


def predict_single_patient(patient_data):
    """Predict for a single patient - uses model"""
    if model is None:
        return None, None
    
    try:
        required_cols = [c for c in df_ml_data.columns if c != TARGET_COL]
        pred_row = {}
        for col in required_cols:
            if col in patient_data and patient_data[col] is not None:
                pred_row[col] = patient_data[col]
            else:
                if df_ml_data[col].dtype in ['int64', 'float64']:
                    pred_row[col] = float(df_ml_data[col].median())
                else:
                    pred_row[col] = df_ml_data[col].mode()[0] if len(df_ml_data[col].mode()) > 0 else 0
        
        df = pd.DataFrame([pred_row])[required_cols]
        prob = model.predict_proba(df)[0][1]
        tier = get_risk_tier(prob)
        return float(prob), tier
    except:
        return None, None


def generate_explanation(vitals, labs, risk_tier, patient_id=None):
    """Generate clinical explanation"""
    all_findings = []
    
    spo2 = vitals.get('spo2_pct')
    if spo2 is not None:
        if spo2 < 75:
            all_findings.append(("profound hypoxemia", "respiratory failure"))
        elif spo2 < 85:
            all_findings.append(("severe hypoxemia", "respiratory failure"))
        elif spo2 < 90:
            all_findings.append(("moderate hypoxemia", "respiratory compromise"))
        elif spo2 < 94:
            all_findings.append(("mild hypoxemia", "impaired oxygenation"))

    rr = vitals.get('respiratory_rate')
    if rr:
        if rr > 30:
            all_findings.append(("severe tachypnea", "respiratory failure"))
        elif rr > 25:
            all_findings.append(("marked tachypnea", "respiratory distress"))
        elif rr > 20:
            all_findings.append(("elevated respiratory rate", "increased work of breathing"))
    
    hr = vitals.get('heart_rate')
    if hr:
        if hr > 130:
            all_findings.append(("severe tachycardia", "cardiovascular instability"))
        elif hr > 110:
            all_findings.append(("tachycardia", "cardiovascular stress"))
        elif hr > 100:
            all_findings.append(("elevated heart rate", "compensatory response"))
    
    sbp = vitals.get('systolic_bp')
    if sbp is not None:
        if sbp < 80:
            all_findings.append(("severe hypotension", "circulatory shock"))
        elif sbp < 90:
            all_findings.append(("hypotension", "hemodynamic instability"))
    
    temp = vitals.get('temperature_c')
    if temp:
        if temp > 39:
            all_findings.append(("high fever", "systemic inflammatory response"))
        elif temp < 36:
            all_findings.append(("hypothermia", "severe sepsis"))
    
    lactate = labs.get('lactate')
    if lactate is not None:
        if lactate > 4:
            all_findings.append(("elevated lactate", "tissue hypoperfusion"))
        elif lactate > 2:
            all_findings.append(("rising lactate", "early tissue hypoxia"))
    
    shock_trigger = ((sbp is not None and sbp < 80) or (lactate is not None and lactate > 4))
    
    wbc = labs.get('wbc_count')
    if wbc:
        if wbc > 20:
            all_findings.append(("severe leukocytosis", "overwhelming infection"))
        elif wbc > 12:
            all_findings.append(("leukocytosis", "inflammatory response"))
    
    sepsis = labs.get('sepsis_risk_score')
    if sepsis and sepsis > 0.5:
        all_findings.append(("elevated sepsis markers", "sepsis-like decompensation"))
    
    # Shuffle slightly based on patient_id
    if patient_id and len(all_findings) > 2:
        np.random.seed(patient_id)
        np.random.shuffle(all_findings)
    
    # If no strong findings, keep it calm especially for low tiers
    strong = any(x[0] in ["severe hypoxemia","marked tachypnea","severe tachypnea"] for x in all_findings)

    if not all_findings:
        exp = "Vital signs are within expected ranges; deterioration risk appears low."
    elif risk_tier in ["MINIMAL", "LOW"] and not strong:
        exp = "Some values warrant monitoring, but there are no strong signals of imminent respiratory failure."
    else:
        # keep the original strong templates only when justified
        if len(all_findings) >= 3:
            f1, f2, f3 = all_findings[:3]
            exp = f"{f1[0].capitalize()} combined with {f2[0]} in the setting of {f3[0]} suggests clinical deterioration risk."
        elif len(all_findings) == 2:
            f1, f2 = all_findings[:2]
            exp = f"{f1[0].capitalize()} together with {f2[0]} suggests elevated deterioration risk."
        else:
            f1 = all_findings[0]
            exp = f"{f1[0].capitalize()} warrants close monitoring."

    if not shock_trigger:
        exp = exp.replace(" shock", "").replace(" and shock", "").replace(" or shock", "")

    return f"Causal explanation: {exp}"



def precompute_all_predictions():
    """Pre-compute predictions for patients at startup"""
    global all_predictions_cache, kpis_cache
    
    print("Pre-computing predictions...")
    
    all_predictions = []
    high_risk_count = 0
    critical_count = 0
    max_risk_patient = None
    
    # Process only 500 patients for fast startup
    patients_to_process = df_patients.head(500)
    total = len(patients_to_process)
    
    for idx, (_, row) in enumerate(patients_to_process.iterrows()):
        pid = int(row['patient_id'])
        
        vitals = get_patient_latest_vitals(pid)
        labs = get_patient_latest_labs(pid)
        patient_info = row.to_dict()
        
        full_data = {**patient_info, **vitals, **labs}
        prob, tier = predict_single_patient(full_data)
        
        if prob is None:
            continue
        
        if tier in ['HIGH', 'CRITICAL']:
            high_risk_count += 1
        if tier == 'CRITICAL':
            critical_count += 1
        if max_risk_patient is None or prob > max_risk_patient.get('probability', 0):
            max_risk_patient = {'patient_id': pid, 'probability': prob, 'risk_tier': tier}
        
        explanation = generate_explanation(vitals, labs, tier, pid)
        
        np.random.seed(pid)
        unit = UNITS[np.random.randint(0, len(UNITS))]
        
        all_predictions.append({
            'patient_id': pid,
            'probability': prob,
            'risk_tier': tier,
            'explanation': explanation,
            'unit': unit,
            'room': f"R{(pid % 20) + 1:02d}",
            'bed': f"B{(pid % 4) + 1}",
        })
    
    all_predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    all_predictions_cache = all_predictions
    
    # Extrapolate KPIs for full dataset
    ratio = len(df_patients) / len(patients_to_process)
    kpis_cache = {
        'patients_in_monitoring': len(df_patients),
        'high_risk_next_12h': int(high_risk_count * ratio),
        'critical_count': int(critical_count * ratio),
        'max_risk': max_risk_patient
    }
    
    tier_counts = {}
    for p in all_predictions:
        t = p['risk_tier']
        tier_counts[t] = tier_counts.get(t, 0) + 1
    
    print(f"Done! {len(all_predictions)} patients in {total} processed")
    print(f"  High-Risk: {kpis_cache['high_risk_next_12h']} (estimated for 10k)")
    for t in ['CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'MINIMAL']:
        print(f"  {t}: {tier_counts.get(t, 0)}")


def load_system():
    """Load all data and model"""
    global df_ml_data, df_patients, df_vitals, df_labs, model, vitals_index, labs_index
    
    print("Loading system...")
    
    if os.path.exists(DATA_PATH):
        df_ml_data = pd.read_csv(DATA_PATH, sep=';')
        print(f"  ML data: {len(df_ml_data)} records")
    
    if os.path.exists(PATIENTS_PATH):
        df_patients = pd.read_csv(PATIENTS_PATH)
        print(f"  Patients: {len(df_patients)}")
    
    if os.path.exists(VITALS_PATH):
        df_vitals = pd.read_csv(VITALS_PATH, sep=';')
        print(f"  Vitals: {len(df_vitals)} records")
        # Build index for fast lookup
        print("  Building vitals index...")
        for pid in df_vitals['patient_id'].unique():
            patient_rows = df_vitals[df_vitals['patient_id'] == pid]
            latest_idx = patient_rows['hour_from_admission'].idxmax()
            vitals_index[pid] = latest_idx
    
    if os.path.exists(LABS_PATH):
        df_labs = pd.read_csv(LABS_PATH, sep=';')
        print(f"  Labs: {len(df_labs)} records")
        # Build index for fast lookup
        print("  Building labs index...")
        for pid in df_labs['patient_id'].unique():
            patient_rows = df_labs[df_labs['patient_id'] == pid]
            latest_idx = patient_rows['hour_from_admission'].idxmax()
            labs_index[pid] = latest_idx
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("  Model loaded")
    
    # Pre-compute all predictions
    precompute_all_predictions()
    
    print("\nSystem ready!")


GEMINI_MODEL = "gemini-2.5-flash"

def call_gemini_api(prompt: str):
    key = os.environ.get("GEMINI_API_KEY","").strip()
    if not key:
        return None
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(api_version="v1")  # مهم: v1 بدل v1beta :contentReference[oaicite:3]{index=3}
    )
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return (resp.text or "").strip() or None


# ============ API ROUTES ============

@app.route('/')
def home():
    return send_file('index.html')


@app.route('/patient_details')
def patient_details_page():
    return send_file('patient_details.html')


@app.route('/api/dashboard_summary')
def dashboard_summary():
    """Dashboard KPIs - uses pre-computed cache"""
    return jsonify({
        "success": True,
        "kpis": {
            "patients_in_monitoring": kpis_cache.get('patients_in_monitoring', 0),
            "high_risk_next_12h": kpis_cache.get('high_risk_next_12h', 0),
            "new_alerts_last_1h": kpis_cache.get('critical_count', 0),
            "max_risk": kpis_cache.get('max_risk')
        },
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/patients_overview')
def patients_overview():
    """Get patients - uses pre-computed cache"""
    limit = min(int(request.args.get('limit', 200)), 200)
    
    # Get proportional from each tier
    per_tier = limit // 5
    
    tiers = {'CRITICAL': [], 'HIGH': [], 'MODERATE': [], 'LOW': [], 'MINIMAL': []}
    for p in all_predictions_cache:
        tiers[p['risk_tier']].append(p)
    
    result = []
    for tier_name in ['CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'MINIMAL']:
        result.extend(tiers[tier_name][:per_tier])
    
    result.sort(key=lambda x: x['probability'], reverse=True)
    
    return jsonify({
        "success": True,
        "patients": result[:limit],
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/patient_details/<int:patient_id>')
def patient_details(patient_id):
    """Get patient details with timeline"""
    try:
        patient_row = df_patients[df_patients['patient_id'] == patient_id]
        if patient_row.empty:
            return jsonify({"success": False, "error": "Patient not found"})
        
        patient_info = patient_row.iloc[0].to_dict()
        
        # Get timelines
        vitals_timeline = []
        if df_vitals is not None:
            pv = df_vitals[df_vitals['patient_id'] == patient_id].sort_values('hour_from_admission').tail(12)
            for _, r in pv.iterrows():
                vitals_timeline.append({
                    'hour': int(r['hour_from_admission']),
                    'hr': round(r['heart_rate'], 0) if pd.notna(r['heart_rate']) else None,
                    'rr': round(r['respiratory_rate'], 0) if pd.notna(r['respiratory_rate']) else None,
                    'spo2': round(r['spo2_pct'], 0) if pd.notna(r['spo2_pct']) else None,
                    'temp': round(r['temperature_c'], 1) if pd.notna(r['temperature_c']) else None,
                    'sbp': round(r['systolic_bp'], 0) if pd.notna(r['systolic_bp']) else None,
                    'dbp': round(r['diastolic_bp'], 0) if pd.notna(r['diastolic_bp']) else None,
                    'o2_flow': round(r['oxygen_flow'], 1) if pd.notna(r['oxygen_flow']) else None,
                })
        
        labs_timeline = []
        if df_labs is not None:
            pl = df_labs[df_labs['patient_id'] == patient_id].sort_values('hour_from_admission').tail(12)
            for _, r in pl.iterrows():
                labs_timeline.append({
                    'hour': int(r['hour_from_admission']),
                    'wbc': round(r['wbc_count'], 2) if pd.notna(r['wbc_count']) else None,
                    'lactate': round(r['lactate'], 2) if pd.notna(r['lactate']) else None,
                    'creatinine': round(r['creatinine'], 2) if pd.notna(r['creatinine']) else None,
                    'hemoglobin': round(r['hemoglobin'], 2) if pd.notna(r['hemoglobin']) else None,
                    'sepsis_score': round(r['sepsis_risk_score'], 3) if pd.notna(r['sepsis_risk_score']) else None,
                })
        
        # Find in cache
        pred = next((p for p in all_predictions_cache if p['patient_id'] == patient_id), None)
        if not pred:
            vitals = get_patient_latest_vitals(patient_id)
            labs = get_patient_latest_labs(patient_id)
            prob, tier = predict_single_patient({**patient_info, **vitals, **labs})
            pred = {'probability': prob or 0, 'risk_tier': tier or 'UNKNOWN'}
        
        return jsonify({
            "success": True,
            "patient": {"patient_id": patient_id, "age": patient_info.get('age'), "gender": patient_info.get('gender')},
            "prediction": pred,
            "explanation": pred.get('explanation', ''),
            "vitals_timeline": vitals_timeline,
            "labs_timeline": labs_timeline,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/llm_reason')
def llm_reason():
    patient_id = int(request.args.get('patient_id', 0))
    pred = next((p for p in all_predictions_cache if p['patient_id'] == patient_id), None)
    if pred:
        return jsonify({"success": True, "patient_id": patient_id, "llm_reason": pred.get('explanation', '')})
    return jsonify({"success": False, "error": "Patient not found"})


@app.route('/api/predict_and_explain', methods=['POST'])
def predict_and_explain():
    """Manual prediction"""
    try:
        body = request.get_json(silent=True) or {}
        patient = body.get('patient', {})
        
        if not patient:
            return jsonify({"success": False, "error": "Patient data required"}), 400
        
        # Fix inconsistent oxygen inputs
        if str(patient.get("oxygen_device", "none")).lower() in ["none", "", "null"]:
            patient["oxygen_flow"] = 0
        
        vitals = {k: patient.get(k) for k in ['heart_rate','respiratory_rate','spo2_pct','temperature_c','systolic_bp','diastolic_bp','oxygen_flow']}

        labs = {k: patient.get(k) for k in ['lactate', 'wbc_count', 'creatinine', 'crp_level', 'hemoglobin', 'sepsis_risk_score']}
        
        full_patient = {**patient, **vitals, **labs}
        prob, tier = predict_single_patient(full_patient)
        if prob is None:
            return jsonify({"success": False, "error": "Prediction failed"}), 500

        tier = apply_clinical_override(vitals, labs, tier)
        if prob is None:
            return jsonify({"success": False, "error": "Prediction failed"}), 500
        
        explanation = None
        provider = "local"
        
        if os.environ.get("GEMINI_API_KEY","").strip():
            vitals_text = "\n".join([f"- {k}: {v}" for k, v in vitals.items() if v is not None])
            labs_text = "\n".join([f"- {k}: {v}" for k, v in labs.items() if v is not None])
            prompt = f"""You are a critical-care clinical reasoning assistant.
Write ONE sentence (max 22 words) explaining why this patient is at risk.
Rules:
- Mention 2-3 findings.
- Match tone to tier:
  * MINIMAL: reassuring tone (no immediate concern; continue routine monitoring). Do NOT say “warrants monitoring”.
  * LOW: mild concern (recommend monitoring), avoid shock/failure.
  * MODERATE: mention concern and need for evaluation.
  * HIGH/CRITICAL: you may mention respiratory failure/shock/sepsis if supported by findings.
- Use "combined with" / "in the setting of".
- No specific numbers.
- Avoid generic phrases like "high risk".
Only use the word "failure" if SpO2 < 85 OR RR > 35 OR SBP < 90 OR lactate > 4.
Otherwise use "respiratory distress" or "possible decline".
NO specific numbers. NO generic phrases like "high risk".

Patient: {tier} ({prob*100:.0f}%)
Vitals:
{vitals_text or '(none)'}
Labs:
{labs_text or '(none)'}

Causal explanation:"""
            explanation = call_gemini_api(prompt)
            if explanation:
                provider = "gemini"
        
        if not explanation:
            explanation = generate_explanation(vitals, labs, tier)
        
        return jsonify({
            "success": True,
            "prediction": {"risk_tier": tier},
            "llm_reason": explanation,
            "provider": provider,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/llm_status")
def llm_status():
    key = os.environ.get("GEMINI_API_KEY","").strip()
    if not key:
        return jsonify({"success": False, "provider":"gemini", "api_key_present": False})
    try:
        client = genai.Client(api_key=key, http_options=types.HttpOptions(api_version="v1"))
        _ = client.models.generate_content(model="gemini-2.5-flash", contents="ping")

        return jsonify({"success": True, "provider":"gemini", "api_key_present": True})
    except Exception as e:
        return jsonify({"success": False, "provider":"gemini", "api_key_present": True, "error": str(e)})



if __name__ == '__main__':
    load_system()
    print("\n" + "="*50)
    print("SERAJ - Patient Monitoring System")
    print("="*50)
    print(f"URL: http://127.0.0.1:5000")
    print(f"Groq API: {'Enabled' if GROQ_API_KEY else 'Disabled'}")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
