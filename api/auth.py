# api/auth.py
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel

# Configuration
SECRET_KEY = "votre-cle-secrete-a-changer-en-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 heures
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Modèles
class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    password: str
    role: str  # "medecin" ou "patient"
    full_name: str

class UserLogin(BaseModel):
    username: str
    password: str

class User(BaseModel):
    username: str
    role: str
    full_name: str
    disabled: bool = False

# Base de données simulée (en production, utiliser une vraie DB)
fake_users_db = {
    "dr_martin": {
        "username": "dr_martin",
        "full_name": "Dr. Martin Dupont",
        "role": "medecin",
        "hashed_password": bcrypt.hashpw("medecin123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "disabled": False
    },
    "patient_001": {
        "username": "patient_001",
        "full_name": "Jean Durand",
        "role": "patient",
        "hashed_password": bcrypt.hashpw("patient123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "disabled": False,
        "patient_id": "ID00007637O"
    },
    "patient_002": {
        "username": "patient_002",
        "full_name": "Marie Lambert",
        "role": "patient",
        "hashed_password": bcrypt.hashpw("patient123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "disabled": False,
        "patient_id": "ID00419637O"
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def get_user(username: str) -> Optional[dict]:
    if username in fake_users_db:
        return fake_users_db[username]
    return None

def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiants invalides",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=payload.get("role"))
    except JWTError:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Utilisateur désactivé")
    return current_user

def require_role(required_role: str):
    async def role_checker(current_user: dict = Depends(get_current_active_user)):
        if current_user["role"] != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Accès réservé aux {required_role}s"
            )
        return current_user
    return role_checker


