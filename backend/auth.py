# backend/auth.py - VERSION CORRIGÉE
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production-1234567890"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Base de données factice avec les bons rôles
fake_users_db: Dict[str, Dict[str, Any]] = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "role": "admin",
        "disabled": False,
    },
    "doctor": {
        "username": "doctor",
        "full_name": "Doctor Smith",
        "email": "doctor@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "role": "medecin",
        "disabled": False,
    },
    "patient1": {
        "username": "patient1",
        "full_name": "Patient One",
        "email": "patient1@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "role": "patient",
        "patient_id": "IDPATIENT1",
        "disabled": False,
    },
    "patient2": {
        "username": "patient2",
        "full_name": "Patient Two",
        "email": "patient2@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "role": "patient",
        "patient_id": "IDPATIENT2",
        "disabled": False,
    }
}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Modèles Pydantic
class Token(BaseModel):
    access_token: str
    token_type: str
    user: Optional[Dict[str, Any]] = None

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    disabled: Optional[bool] = None
    patient_id: Optional[str] = None

class UserInDB(User):
    hashed_password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str
    password: str
    role: str = "patient"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def get_user(db: Dict[str, Dict[str, Any]], username: str) -> Optional[UserInDB]:
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(db: Dict[str, Dict[str, Any]], username: str, password: str) -> Optional[UserInDB]:
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
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

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception
    
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def require_role(required_role: str):
    async def role_checker(current_user: User = Depends(get_current_active_user)):
        if current_user.role != required_role:
            raise HTTPException(
                status_code=403,
                detail=f"Requires {required_role} role. Current role: {current_user.role}"
            )
        return current_user
    return role_checker
