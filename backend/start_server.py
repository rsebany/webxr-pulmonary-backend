# backend/start_server.py
import uvicorn

if __name__ == "__main__":
    print("🚀 Démarrage de l'API Pulmonary Fibrosis...")
    print("📡 URL: http://localhost:8000")
    print("📋 Documentation: http://localhost:8000/docs")
    print("⏹️  Arrêt: Ctrl+C")
    print("-" * 50)
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Redémarrage auto sur changement de code
        log_level="info"
    )