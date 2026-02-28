from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
import subprocess
import datetime
import firebase_admin
from firebase_admin import credentials, firestore

app = FastAPI(title="一口馬主 AI予測システム API")

# Initialize Firebase Admin SDK
# On Google Cloud Run, default credentials will be used automatically.
# For local testing, ensure GOOGLE_APPLICATION_CREDENTIALS points to a service account key JSON file if needed,
# or simply let it initialize without credentials if your local emulator handles it.
try:
    firebase_admin.initialize_app(options={'projectId': 'jra-horse-prediction-9224f'})
    db = firestore.client()
    print("Firebase initialized successfully for target project: jra-horse-prediction-9224f")
except Exception as e:
    print(f"Warning: Could not initialize Firebase. Error: {e}")
    db = None


# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define base_dir dynamically based on the current file's location
base_dir = os.path.dirname(os.path.abspath(__file__))
model_tier_path = os.path.join(base_dir, 'models', 'lgbm_model_tier.pkl')
model_surface_path = os.path.join(base_dir, 'models', 'lgbm_model_surface.pkl')
encoder_path = os.path.join(base_dir, 'models', 'encoder.pkl')
features_path = os.path.join(base_dir, 'data', 'processed', 'processed_features.csv')

# Global variables to hold model and encoder
model_tier = None
model_surface = None
encoder = None
options_cache = {}
sire_stats_cache = {}
comb_stats_cache = {}
env_stats_cache = {}
global_defaults = {}

def load_artifacts():
    global model_tier, model_surface, encoder, options_cache
    try:
        model_tier = joblib.load(model_tier_path)
        encoder = joblib.load(encoder_path)
        print("Tier model and encoder loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load Tier model or encoder. Error: {e}")
        
    try:
        if os.path.exists(model_surface_path):
            model_surface = joblib.load(model_surface_path)
            print("Surface model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load Surface model. Error: {e}")

    try:
        global sire_stats_cache, comb_stats_cache, env_stats_cache
        if os.path.exists(features_path):
            df = pd.read_csv(features_path)
            opts = {}
            for col in ['厩舎', '馬主', '生産者', '生産地', '種牡馬', '母父名', '父タイプ名', '母父タイプ名']:
                if col in df.columns:
                    # Get unique non-null values, sorted
                    unique_vals = [str(x) for x in df[col].dropna().unique() if str(x) != 'nan']
                    opts[col] = sorted(unique_vals)
            
            # Create linkage dictionaries
            sire_to_line = {}
            if '種牡馬' in df.columns and '父タイプ名' in df.columns:
                mapping = df.dropna(subset=['種牡馬', '父タイプ名']).drop_duplicates(subset=['種牡馬'])
                sire_to_line = dict(zip(mapping['種牡馬'].astype(str), mapping['父タイプ名'].astype(str)))
                
            bm_sire_to_line = {}
            if '母父名' in df.columns and '母父タイプ名' in df.columns:
                mapping = df.dropna(subset=['母父名', '母父タイプ名']).drop_duplicates(subset=['母父名'])
                bm_sire_to_line = dict(zip(mapping['母父名'].astype(str), mapping['母父タイプ名'].astype(str)))
                
            breeder_to_location = {}
            if '生産者' in df.columns and '生産地' in df.columns:
                mapping = df.dropna(subset=['生産者', '生産地']).drop_duplicates(subset=['生産者'])
                breeder_to_location = dict(zip(mapping['生産者'].astype(str), mapping['生産地'].astype(str)))
                
            opts['sire_map'] = sire_to_line
            opts['bm_sire_map'] = bm_sire_to_line
            opts['breeder_map'] = breeder_to_location
            options_cache = opts

            # Extract Inference Caches for Win Rates
            sire_df = df.drop_duplicates(subset=['種牡馬'])
            for _, row in sire_df.iterrows():
                sire_stats_cache[str(row['種牡馬'])] = {
                    'Sire_Turf_WinRate': float(row['Sire_Turf_WinRate']),
                    'Sire_Dirt_WinRate': float(row['Sire_Dirt_WinRate']),
                    'Sire_Avg_Win_Dist': float(row['Sire_Avg_Win_Dist'])
                }

            comb_df = df.drop_duplicates(subset=['種牡馬', '母父名'])
            for _, row in comb_df.iterrows():
                comb_key = f"{row['種牡馬']}_{row['母父名']}"
                comb_stats_cache[comb_key] = {
                    'Comb_Turf_WinRate': float(row['Comb_Turf_WinRate']),
                    'Comb_Dirt_WinRate': float(row['Comb_Dirt_WinRate']),
                    'Comb_Avg_Win_Dist': float(row['Comb_Avg_Win_Dist'])
                }

            env_df = df.drop_duplicates(subset=['厩舎', '馬主'])
            for _, row in env_df.iterrows():
                env_key = f"{row['厩舎']}_{row['馬主']}"
                env_stats_cache[env_key] = {
                    'Env_Turf_WinRate': float(row['Env_Turf_WinRate']),
                    'Env_Dirt_WinRate': float(row['Env_Dirt_WinRate']),
                    'Env_Avg_Win_Dist': float(row['Env_Avg_Win_Dist'])
                }

            # Phase 10: Calculate global averages to use as fallback for unseen data
            global global_defaults
            global_defaults = df[[
                'Sire_Turf_WinRate', 'Sire_Dirt_WinRate', 'Sire_Avg_Win_Dist',
                'Comb_Turf_WinRate', 'Comb_Dirt_WinRate', 'Comb_Avg_Win_Dist',
                'Env_Turf_WinRate', 'Env_Dirt_WinRate', 'Env_Avg_Win_Dist'
            ]].mean().to_dict()

            print("Options and Stats cache loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load options or stats cache. Error: {e}")

# Initial load
load_artifacts()

categorical_cols = ['性別', '厩舎', '馬主', '生産者', '生産地', '種牡馬', '母父名', '父タイプ名', '母父タイプ名']
numerical_cols = [
    'BirthMonth', 
    'Sire_Turf_WinRate', 'Sire_Dirt_WinRate', 'Sire_Avg_Win_Dist',
    'Comb_Turf_WinRate', 'Comb_Dirt_WinRate', 'Comb_Avg_Win_Dist',
    'Env_Turf_WinRate', 'Env_Dirt_WinRate', 'Env_Avg_Win_Dist'
]

class PredictionRequest(BaseModel):
    性別: str
    BirthMonth: int
    厩舎: str
    馬主: str
    生産者: str
    生産地: str
    種牡馬: str
    母父名: str
    父タイプ名: str
    母父タイプ名: str

@app.get("/")
def read_index():
    index_path = os.path.join(base_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found"}

@app.get("/retrain-model")
@app.get("/retrain-model.html")
def read_retrain_page():
    retrain_path = os.path.join(base_dir, "secret", "retrain-model.html")
    if os.path.exists(retrain_path):
        return FileResponse(retrain_path)
    return {"error": "retrain-model.html not found"}

@app.get("/options")
def get_options():
    return options_cache

def log_prediction_to_firestore(request_info: dict, input_data: dict, prediction_result: dict):
    if db is None:
        return
    try:
        doc_ref = db.collection('prediction_logs').document()
        log_entry = {
            "timestamp": firestore.SERVER_TIMESTAMP, # Or datetime.datetime.now(datetime.timezone.utc)
            "request_info": request_info,
            "input_data": input_data,
            "prediction_result": prediction_result
        }
        doc_ref.set(log_entry)
        print("Logged prediction to Firestore.")
    except Exception as e:
        print(f"Failed to log to Firestore: {e}")

@app.post("/predict")
def predict_tier(req: PredictionRequest, request: Request, background_tasks: BackgroundTasks):

    if model_tier is None or encoder is None:
        raise HTTPException(status_code=503, detail="Tier Model is not loaded.")
    try:
        # Construct DataFrame
        if hasattr(req, 'model_dump'):
            input_data = req.model_dump()
        else:
            input_data = req.dict()
            
        df = pd.DataFrame([input_data])
        
        # Inject Dynamic Win Rates from Cache
        sire_name = str(input_data.get('種牡馬', ''))
        bmsire_name = str(input_data.get('母父名', ''))
        comb_key = f"{sire_name}_{bmsire_name}"
        stable_name = str(input_data.get('厩舎', ''))
        owner_name = str(input_data.get('馬主', ''))
        env_key = f"{stable_name}_{owner_name}"
        
        # Fallback defaults if components are completely unseen
        default_sire = {
            'Sire_Turf_WinRate': global_defaults.get('Sire_Turf_WinRate', 0.1),
            'Sire_Dirt_WinRate': global_defaults.get('Sire_Dirt_WinRate', 0.1),
            'Sire_Avg_Win_Dist': global_defaults.get('Sire_Avg_Win_Dist', 1600.0)
        }
        default_comb = {
            'Comb_Turf_WinRate': global_defaults.get('Comb_Turf_WinRate', 0.1),
            'Comb_Dirt_WinRate': global_defaults.get('Comb_Dirt_WinRate', 0.1),
            'Comb_Avg_Win_Dist': global_defaults.get('Comb_Avg_Win_Dist', 1600.0)
        }
        default_env = {
            'Env_Turf_WinRate': global_defaults.get('Env_Turf_WinRate', 0.1),
            'Env_Dirt_WinRate': global_defaults.get('Env_Dirt_WinRate', 0.1),
            'Env_Avg_Win_Dist': global_defaults.get('Env_Avg_Win_Dist', 1600.0)
        }
        
        # Track if we are using historical data or global average fallbacks
        sire_known = sire_name in sire_stats_cache
        env_known = env_key in env_stats_cache
        
        sire_vals = sire_stats_cache.get(sire_name, default_sire)
        comb_vals = comb_stats_cache.get(comb_key, default_comb)
        env_vals = env_stats_cache.get(env_key, default_env)
        
        for k, v in sire_vals.items():
            df[k] = v
        for k, v in comb_vals.items():
            df[k] = v
        for k, v in env_vals.items():
            df[k] = v
            
        # Encoding categoricals
        for col in categorical_cols:
            if col not in df.columns:
                df[col] = 'Unknown'
            df[col] = df[col].astype(str).fillna('Unknown')
            
        df_encoded = df.copy()
        df_encoded[categorical_cols] = encoder.transform(df[categorical_cols])
        
        for col in categorical_cols:
            df_encoded[col] = df_encoded[col].astype(int)
            
        # Ensure correct column order
        X = df_encoded[categorical_cols + numerical_cols]
        
        # Predict Tier
        probs_tier = model_tier.predict_proba(X)[0]
        pred_class_tier = int(model_tier.predict(X)[0])
        
        classes_tier = ["Un-raced", "1 win", "2 wins", "3 wins", "Open class (4+ wins)"]
        
        # Predict Surface
        surface_pred = "Unknown"
        surface_probs = {"Turf": 0.0, "Dirt": 0.0}
        if model_surface is not None:
            probs_surf = model_surface.predict_proba(X)[0]
             
            # Phase 13: Adjust Surface Prediction Sire Weight (30% Sire / 70% Model)
            final_prob_turf = probs_surf[0]
            final_prob_dirt = probs_surf[1]
             
            if sire_known:
                sire_turf = sire_vals.get('Sire_Turf_WinRate', 0)
                sire_dirt = sire_vals.get('Sire_Dirt_WinRate', 0)
                total_sire_win = sire_turf + sire_dirt
                 
                if total_sire_win > 0:
                    sire_ratio_turf = sire_turf / total_sire_win
                    sire_ratio_dirt = sire_dirt / total_sire_win
                     
                    # Blend 70% AI Model + 30% Pure Sire Ratio
                    final_prob_turf = (probs_surf[0] * 0.70) + (sire_ratio_turf * 0.30)
                    final_prob_dirt = (probs_surf[1] * 0.70) + (sire_ratio_dirt * 0.30)
                     
            if final_prob_turf >= final_prob_dirt:
                surface_pred = "Turf"
            else:
                surface_pred = "Dirt"
                 
            surface_probs = {"Turf": float(final_prob_turf), "Dirt": float(final_prob_dirt)}
             
        # Generate Qualitative Reasons (Phase 8)
        reasons = []
        
        # 1. Sire Aptitude
        if sire_known:
            sire_turf = sire_vals.get('Sire_Turf_WinRate', 0)
            sire_dirt = sire_vals.get('Sire_Dirt_WinRate', 0)
            sire_dist = sire_vals.get('Sire_Avg_Win_Dist', 0)
            if sire_turf > sire_dirt + 0.05:
                reasons.append(f"種牡馬{sire_name}は、芝でのレースに高い適性を示すデータが出ています。")
            elif sire_dirt > sire_turf + 0.05:
                reasons.append(f"種牡馬{sire_name}の産駒は、ダート戦を得意とする傾向が強いです。")
                
            if sire_dist > 0:
                if sire_dist < 1400:
                    reasons.append("短距離指向のスピードに優れた血統背景を持ちます。")
                elif sire_dist > 2000:
                    reasons.append("中長距離に適したスタミナ豊かな血統背景を持ちます。")
        else:
            if sire_name and sire_name != "Unknown":
                reasons.append("血統の過去データが少ないため、一般的な平均値をベースにポテンシャルを推測しています。")
                
        # 2. Stable + Owner Synergy
        if env_known:
            env_turf = env_vals.get('Env_Turf_WinRate', 0)
            env_dirt = env_vals.get('Env_Dirt_WinRate', 0)
            # Check general stable success and specific combo success
            if env_turf > 0.08 or env_dirt > 0.08:
                reasons.append(f"{stable_name}厩舎の管理馬は、過去のデータから堅実な好成績を残しています。")
            if env_turf > 0.12 or env_dirt > 0.12:
                reasons.append(f"馬主「{owner_name}」と「{stable_name}」厩舎の組み合わせは、特筆すべき高い期待値を持っています。")
        else:
            if stable_name and stable_name != "Unknown":
                reasons.append("厩舎・馬主に関するより詳細な実績データがないため、未知数の変数を含んだ予測となっています。")
            
        # 3. Bloodline Synergy
        if sire_known:
            comb_turf = comb_vals.get('Comb_Turf_WinRate', 0)
            comb_dirt = comb_vals.get('Comb_Dirt_WinRate', 0)
            if comb_turf > 0.08 or comb_dirt > 0.08:
                reasons.append(f"父{sire_name} × 母父{bmsire_name} の配合は、データ上良好な相性を示しています。")
            
        # 4. Birth Month
        birth_month = input_data.get('BirthMonth', 0)
        try:
            bm = int(birth_month)
            if bm <= 3 and bm != 0:
                reasons.append("早生まれであり、早期からのレースへの適応力にアドバンテージがあります。")
            elif bm >= 5:
                reasons.append("遅生まれの傾向がありますが、今後の大きな成長力が期待できるプロフィールです。")
        except:
            pass
            
        if len(reasons) == 0:
            reasons.append("血統や厩舎の総合的な傾向から、フラットな視点でAI判定を行いました。")
            
        # Phase 12: Predict Distance Aptitude
        sire_dist = sire_vals.get('Sire_Avg_Win_Dist', 1600.0)
        comb_dist = comb_vals.get('Comb_Avg_Win_Dist', 1600.0)
        
        expected_dist = 1600.0
        if sire_known and comb_dist > 0:
            expected_dist = (sire_dist + comb_dist) / 2
        elif sire_known:
            expected_dist = sire_dist
        else:
            expected_dist = global_defaults.get('Sire_Avg_Win_Dist', 1600.0)
            
        dist_cat = ""
        if expected_dist < 1400:
            dist_cat = "短距離"
        elif expected_dist < 1800:
            dist_cat = "マイル"
        elif expected_dist < 2400:
            dist_cat = "中距離"
        else:
            dist_cat = "ステイヤー"
            
        distance_prediction = f"この馬は{dist_cat}適性が高い可能性があります。"
        
        result = {
            "prediction": classes_tier[pred_class_tier],
            "prediction_tier": pred_class_tier,
            "probabilities": {classes_tier[i]: float(probs_tier[i]) for i in range(5)},
            "surface_prediction": surface_pred,
            "surface_probabilities": surface_probs,
            "distance_prediction": distance_prediction,
            "reasons": reasons
        }
        
        # Capture request info for logging
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain a comma-separated list of IPs. The first one is the original client.
            client_ip = forwarded_for.split(",")[0].strip()
            
        request_info = {
            "ip_address": client_ip,
            "user_agent": request.headers.get("User-Agent", "Unknown")
        }
        
        # Run the logging synchronously to prevent Cloud Run from throttling CPU before the gRPC request finishes
        log_prediction_to_firestore(request_info, input_data.copy(), result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_retrain_pipeline():
    try:
        pipeline_dir = os.path.join(base_dir, "training_pipeline")
        print("Starting feature extraction...")
        # Run build_features.py
        subprocess.run(["python", "build_features.py"], cwd=pipeline_dir, check=True)
        print("Feature extraction complete. Starting model training...")
        # Run train_model.py
        subprocess.run(["python", "train_model.py"], cwd=pipeline_dir, check=True)
        print("Model training complete. Reloading artifacts in memory...")
        load_artifacts()
        print("Retraining pipeline finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during retraining pipeline: {e}")
    except Exception as e:
        print(f"Unexpected error during retraining: {e}")

@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_retrain_pipeline)
    return {"message": "Retraining started in the background. It will reload automatically when finished."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
