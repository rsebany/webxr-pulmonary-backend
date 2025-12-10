"""
Vercel Serverless Function wrapper for FastAPI
"""
# Importer l'application FastAPI depuis le même répertoire
from fastapi_app import app
from mangum import Mangum

# Wrapper Mangum pour adapter FastAPI à AWS Lambda/Vercel
# Vercel détecte automatiquement cette variable 'handler'
handler = Mangum(app, lifespan="off")

# Alternative: exporter aussi 'app' directement si nécessaire
# Vercel peut utiliser soit 'handler' soit 'app'
__all__ = ["handler", "app"]

