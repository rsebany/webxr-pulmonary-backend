#!/usr/bin/python3.11
"""
WSGI configuration for FastAPI on PythonAnywhere
Corrected version using asgiref for ASGI-to-WSGI conversion
"""

import sys
import os

# ==================== CONFIGURATION ====================
# Adjust this path to match your PythonAnywhere setup
APP_PATH = '/home/rsebany/webxr-pulmonary-backend/backend'

# Add the app path to Python path if not already there
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

# Change to the app directory
os.chdir(APP_PATH)

print(f"🚀 PythonAnywhere - FastAPI Pulmonary Fibrosis API")
print(f"📂 Loading from: {APP_PATH}")
print(f"📁 Current dir: {os.getcwd()}")

# ==================== CREATE WSGI APPLICATION ====================
try:
    # Import your FastAPI app
    from fastapi_app import app
    
    print(f"✅ FastAPI application loaded: {app.title}")
    
    # Convert ASGI app to WSGI using asgiref (standard method)
    try:
        from asgiref.wsgi import WsgiToAsgi
        
        # Wrap the FastAPI ASGI app with WSGI adapter
        application = WsgiToAsgi(app)
        print("✅ Using asgiref.wsgi.WsgiToAsgi for ASGI-to-WSGI conversion")
        
    except ImportError:
        print("⚠️  asgiref not found. Installing: pip install asgiref")
        print("⚠️  Falling back to alternative method...")
        
        # Alternative: Try using mangum (if available)
        try:
            from mangum import Mangum
            application = Mangum(app)
            print("✅ Using mangum.Mangum adapter")
        except ImportError:
            print("❌ No ASGI-to-WSGI adapter available")
            print("📦 Please install: pip install asgiref")
            
            # Minimal error handler
            def application(environ, start_response):
                status = '500 Internal Server Error'
                headers = [('Content-type', 'application/json')]
                start_response(status, headers)
                error_msg = {
                    "error": "ASGI adapter not installed",
                    "message": "Please install asgiref: pip install asgiref",
                    "status": "error"
                }
                import json
                return [json.dumps(error_msg).encode('utf-8')]
    
    print(f"\n🎉 Application ready for PythonAnywhere!")
    print(f"🌐 Access at: https://rsebany.pythonanywhere.com/")
    print(f"📚 Docs at: https://rsebany.pythonanywhere.com/docs")
    print(f"💚 Health check: https://rsebany.pythonanywhere.com/health")

except ImportError as e:
    print(f"\n❌ ImportError: {e}")
    import traceback
    traceback.print_exc()
    
    # Create a simple error handler
    def application(environ, start_response):
        status = '500 Internal Server Error'
        headers = [('Content-type', 'application/json')]
        start_response(status, headers)
        import json
        error_msg = {
            "status": "error",
            "error": "ImportError",
            "message": str(e),
            "path": APP_PATH,
            "suggestion": "Check that fastapi_app.py exists and all dependencies are installed"
        }
        return [json.dumps(error_msg).encode('utf-8')]

except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    
    # Error handling application
    def application(environ, start_response):
        status = '500 Internal Server Error'
        headers = [('Content-type', 'application/json')]
        start_response(status, headers)
        import json
        error_msg = {
            "status": "error",
            "error": "Application failed to start",
            "details": str(e),
            "type": type(e).__name__
        }
        return [json.dumps(error_msg).encode('utf-8')]

print("\n" + "=" * 60)

