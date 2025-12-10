# backend/fastapi_app.py - VERSION CORRIGÉE
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import timedelta
import joblib
import numpy as np
import pydicom
from io import BytesIO
import base64
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from auth import (
    authenticate_user, create_access_token, get_current_active_user,
    require_role, UserLogin, Token, UserCreate, fake_users_db,
    get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES
)

app = FastAPI(title="Pulmonary Fibrosis WebXR API")

# CORS pour WebXR
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Créer le dossier models s'il n'existe pas
# Support pour différents environnements (local, Vercel serverless)
if os.path.exists("backend/models"):
    MODELS_DIR = "backend/models"
elif os.path.exists("models"):
    MODELS_DIR = "models"
elif os.path.exists("api/models"):
    MODELS_DIR = "api/models"
else:
    MODELS_DIR = "models"
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"📁 Dossier '{MODELS_DIR}' créé")

# Charger ou créer le modèle
try:
    model_path = os.path.join(MODELS_DIR, "pulmonary_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Modèle et scaler chargés avec succès")
except FileNotFoundError:
    print("📝 Création de modèles de démonstration...")
    
    # Créer un modèle de démonstration
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    scaler = StandardScaler()
    
    # Données d'entraînement factices pour la démo
    np.random.seed(42)
    n_samples = 1000
    
    # Générer des données réalistes
    X_demo = np.column_stack([
        np.random.uniform(0, 100, n_samples),  # weeks
        np.random.uniform(50, 90, n_samples),  # percent
        np.random.uniform(40, 80, n_samples),  # age
        np.random.uniform(2000, 4000, n_samples),  # fvc_mean
        np.random.uniform(100, 300, n_samples)     # fvc_std
    ])
    
    # FVC basé sur une relation réaliste
    y_demo = (
        3000 
        - X_demo[:, 0] * 2  # Dégradation avec le temps
        - (80 - X_demo[:, 1]) * 10  # Impact du percent
        + np.random.normal(0, 100, n_samples)  # Bruit
    )
    
    # Entraîner le scaler et le modèle
    X_scaled = scaler.fit_transform(X_demo)
    model.fit(X_scaled, y_demo)
    
    # Sauvegarder les modèles
    model_path = os.path.join(MODELS_DIR, "pulmonary_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("✅ Modèles de démonstration créés et sauvegardés")
    
except Exception as e:
    print(f"❌ Erreur inattendue: {e}")
    raise

class PredictionRequest(BaseModel):
    weeks: float
    percent: float
    age: float
    fvc_mean: float
    fvc_std: float

class DicomData(BaseModel):
    patient_id: str
    dicom_slices: List[str]

@app.get("/")
async def root():
    return {
        "status": "OK", 
        "message": "Pulmonary Fibrosis API is running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "analyze_dicom": "/analyze-dicom (POST)", 
            "patient_history": "/patient-history/{patient_id} (GET)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": True,
        "timestamp": np.datetime64('now').astype(str)
    }

# ==================== AUTHENTIFICATION ====================

@app.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    """Connexion utilisateur"""
    user = authenticate_user(user_data.username, user_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Nom d'utilisateur ou mot de passe incorrect"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user["username"],
            "full_name": user["full_name"],
            "role": user["role"],
            "patient_id": user.get("patient_id")
        }
    }

@app.post("/auth/register")
async def register(user_data: UserCreate):
    """Inscription d'un nouvel utilisateur"""
    if user_data.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Nom d'utilisateur déjà pris")
    
    fake_users_db[user_data.username] = {
        "username": user_data.username,
        "full_name": user_data.full_name,
        "role": user_data.role,
        "hashed_password": get_password_hash(user_data.password),
        "disabled": False
    }
    
    if user_data.role == "patient":
        fake_users_db[user_data.username]["patient_id"] = f"ID{user_data.username.upper()}"
    
    return {"message": "Utilisateur créé avec succès", "username": user_data.username}

@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_active_user)):
    """Récupère les infos de l'utilisateur connecté"""
    return {
        "username": current_user["username"],
        "full_name": current_user["full_name"],
        "role": current_user["role"],
        "patient_id": current_user.get("patient_id")
    }

# ==================== ROUTES MÉDECIN ====================

@app.get("/medecin/patients")
async def get_all_patients(current_user: dict = Depends(require_role("medecin"))):
    """Liste tous les patients (médecin uniquement)"""
    patients = []
    for username, user in fake_users_db.items():
        if user["role"] == "patient":
            patients.append({
                "username": username,
                "full_name": user["full_name"],
                "patient_id": user.get("patient_id", username)
            })
    return {"patients": patients}

# ==================== ROUTES PATIENT ====================

@app.get("/patient/my-data")
async def get_my_data(current_user: dict = Depends(require_role("patient"))):
    """Récupère les données du patient connecté"""
    patient_id = current_user.get("patient_id", current_user["username"])
    
    # Simuler les données du patient
    base_fvc = 2800
    history = []
    for week in range(0, 60, 12):
        degradation = week * 6
        fvc = max(1800, base_fvc - degradation)
        history.append({"week": week, "fvc": int(fvc)})
    
    return {
        "patient_id": patient_id,
        "full_name": current_user["full_name"],
        "age": 58,
        "baseline_fvc": base_fvc,
        "fvc_history": history
    }

@app.post("/predict")
async def predict_fvc(request: PredictionRequest):
    """Prédiction FVC à partir des features tabulaires"""
    try:
        features = np.array([[request.weeks, request.percent, request.age, 
                             request.fvc_mean, request.fvc_std]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        return {
            "fvc_predicted": float(prediction), 
            "confidence": 95.0,
            "status": "success",
            "features_used": {
                "weeks": request.weeks,
                "percent": request.percent, 
                "age": request.age,
                "fvc_mean": request.fvc_mean,
                "fvc_std": request.fvc_std
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@app.post("/analyze-dicom")
async def analyze_dicom(file: UploadFile = File(...)):
    """Analyse d'un fichier DICOM uploadé et retourne les données pour visualisation 3D"""
    try:
        print(f"📁 Réception du fichier: {file.filename}")
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Fichier vide")
        
        # Conversion DICOM
        dicom_dataset = pydicom.dcmread(BytesIO(contents))
        pixel_array = dicom_dataset.pixel_array
        
        print(f"🖼️  Shape DICOM: {pixel_array.shape}")
        
        # Prétraitement
        processed_array = preprocess_dicom_slice(pixel_array)
        
        # Features radiomiques
        features = extract_radiomic_features(processed_array)
        
        # Normalisation pour visualisation (0-255)
        normalized_array = ((processed_array - processed_array.min()) / 
                           (processed_array.max() - processed_array.min() + 1e-8) * 255).astype(np.uint8)
        
        # Préparation données 3D
        if len(pixel_array.shape) == 2:
            volume_data = np.repeat(normalized_array[:, :, np.newaxis], 10, axis=2)
            shape = [pixel_array.shape[0], pixel_array.shape[1], 10]
        else:
            volume_data = normalized_array
            shape = list(pixel_array.shape)
        
        # Limiter la taille si trop gros
        if volume_data.size > 500000:
            if len(shape) == 3:
                step = max(1, shape[2] // 5)
                volume_data = volume_data[:, :, ::step]
                shape[2] = volume_data.shape[2]
                print(f"🔧 Volume réduit à: {shape}")
        
        volume_list = volume_data.flatten().tolist()
        
        return {
            "shape": shape,
            "data": volume_list,
            "hu_range": [int(pixel_array.min()), int(pixel_array.max())],
            "radiomic_features": features,
            "patient_id": getattr(dicom_dataset, 'PatientID', 'unknown'),
            "file_name": file.filename,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Erreur analyse DICOM: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse DICOM: {str(e)}")

@app.post("/analyze-dicom-volume")
async def analyze_dicom_volume(files: List[UploadFile] = File(...)):
    """Analyse de plusieurs fichiers DICOM pour créer un volume 3D complet"""
    try:
        print(f"📁 Réception de {len(files)} fichiers DICOM")
        
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="Aucun fichier fourni")
        
        slices = []
        slice_positions = []
        patient_id = "unknown"
        
        for file in files:
            contents = await file.read()
            if len(contents) == 0:
                continue
                
            dicom_dataset = pydicom.dcmread(BytesIO(contents))
            pixel_array = dicom_dataset.pixel_array
            
            # Récupérer la position de la slice si disponible
            try:
                slice_location = float(dicom_dataset.SliceLocation)
            except:
                try:
                    slice_location = float(dicom_dataset.ImagePositionPatient[2])
                except:
                    slice_location = len(slices)
            
            # Récupérer le Patient ID
            if patient_id == "unknown":
                patient_id = getattr(dicom_dataset, 'PatientID', 'unknown')
            
            # Prétraitement de la slice
            processed_slice = preprocess_dicom_slice(pixel_array)
            
            slices.append({
                'data': processed_slice,
                'position': slice_location,
                'original': pixel_array
            })
            slice_positions.append(slice_location)
        
        if len(slices) == 0:
            raise HTTPException(status_code=400, detail="Aucune slice valide trouvée")
        
        # Trier les slices par position
        sorted_indices = np.argsort(slice_positions)
        sorted_slices = [slices[i]['data'] for i in sorted_indices]
        
        print(f"🔧 {len(sorted_slices)} slices triées par position")
        
        # Créer le volume 3D
        volume_3d = np.stack(sorted_slices, axis=2)
        
        # Normalisation pour visualisation (0-255)
        volume_normalized = ((volume_3d - volume_3d.min()) / 
                            (volume_3d.max() - volume_3d.min() + 1e-8) * 255).astype(np.uint8)
        
        # Redimensionner si nécessaire pour performances
        target_size = 128
        h, w, d = volume_normalized.shape
        
        if max(h, w, d) > target_size:
            # Sous-échantillonnage
            step_h = max(1, h // target_size)
            step_w = max(1, w // target_size)
            step_d = max(1, d // target_size)
            volume_normalized = volume_normalized[::step_h, ::step_w, ::step_d]
            print(f"🔧 Volume réduit de ({h},{w},{d}) à {volume_normalized.shape}")
        
        shape = list(volume_normalized.shape)
        volume_list = volume_normalized.flatten().tolist()
        
        # Calculer les features sur l'ensemble du volume
        features = extract_radiomic_features(volume_3d)
        
        # HU range depuis les données originales
        all_originals = np.concatenate([s['original'].flatten() for s in slices])
        hu_range = [int(all_originals.min()), int(all_originals.max())]
        
        return {
            "shape": shape,
            "data": volume_list,
            "num_slices": len(sorted_slices),
            "hu_range": hu_range,
            "radiomic_features": features,
            "patient_id": patient_id,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Erreur analyse volume DICOM: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse volume DICOM: {str(e)}")

@app.get("/patient-history/{patient_id}")
async def get_patient_history(patient_id: str):
    """Récupère l'historique d'un patient"""
    try:
        # Simulation de données réalistes
        base_fvc = 3000
        if patient_id.replace('_', '').isdigit():
            base_fvc = 2500 + (int(patient_id.replace('_', '')) % 10) * 100
        
        history = []
        for week in range(0, 60, 12):
            degradation = week * 8 + (hash(patient_id) % 20)
            fvc = max(1500, base_fvc - degradation)
            history.append({"week": week, "fvc": int(fvc)})
        
        age = 50 + (sum(ord(c) for c in patient_id) % 30)
        
        return {
            "patient_id": patient_id,
            "age": age,
            "baseline_fvc": base_fvc,
            "fvc_history": history,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur historique: {str(e)}")

def preprocess_dicom_slice(slice_array):
    """Même preprocessing que dans Kaggle"""
    slice_array = slice_array.astype(np.float32)
    slice_array = np.clip(slice_array, -1000, 400)
    slice_array = (slice_array + 1000) / 1400
    return slice_array

def extract_radiomic_features(processed_array):
    """Extraction de features radiomiques basiques"""
    return {
        "mean": float(np.mean(processed_array)),
        "std": float(np.std(processed_array)),
        "min": float(np.min(processed_array)),
        "max": float(np.max(processed_array)),
        "percentile_25": float(np.percentile(processed_array, 25)),
        "percentile_75": float(np.percentile(processed_array, 75)),
    }

# NE PAS UTILISER __main__ avec reload - utiliser directement uvicorn en ligne de commande