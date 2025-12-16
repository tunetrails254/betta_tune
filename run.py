from flask import Flask, redirect, url_for, session, request
from flask_cors import CORS
from app.routes import routes
import os
import logging
import sys
from flask_dance.contrib.google import make_google_blueprint, google
from dotenv import load_dotenv

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.WARNING)  # less noise

def create_app():
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    
    # Basic config
    app.secret_key = os.getenv("SECRET_KEY") or os.urandom(24)
    CORS(app)  # allows your frontend to talk to backend

    # Register your routes (lessons, gigs, etc.)
    app.register_blueprint(routes)

    # Google Login Setup
    google_bp = make_google_blueprint(
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scope=[
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/userinfo.email",
            "openid"
        ],
        redirect_url="/login/google/authorized"  # this is the correct way
    )
    app.register_blueprint(google_bp, url_prefix="/login")

    # Google Callback Route (this fixes the 404 error)
    @app.route("/login/google/authorized")
    def google_auth_callback():
        if not google.authorized:
            logger.error("Google login failed")
            return redirect(f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/login?error=google_failed")
        
        resp = google.get("/oauth2/v2/userinfo")
        if resp.ok:
            user_info = resp.json()
            session["user"] = {
                "email": user_info["email"],
                "name": user_info["name"],
                "picture": user_info.get("picture")
            }
            logger.info(f"User logged in: {user_info['email']}")
        
        # Change this URL to your actual frontend page
        return redirect(os.getenv('FRONTEND_URL', 'http://localhost:3000') + "/dashboard")

    return app

app = create_app()

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )