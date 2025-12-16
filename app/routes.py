from flask import Blueprint, json, render_template_string, request, jsonify, render_template, redirect, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import re
from app.utils import extract_features
from app.model import load_assets
from app.config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from app.database import init_db
import pandas as pd
import sqlite3
import os
from datetime import datetime
import logging
import secrets
from flask_dance.contrib.google import google
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)
routes = Blueprint('routes', __name__)

# Load models once
assets_loaded = load_assets()
(
    GENDER_MODELS, SCALER_GENDER, FEATURE_LIST,
    MODEL_STEP1, SCALER_STEP1, LABEL_ENCODER_STEP1,
    MODEL_STEP2, SCALER_STEP2, LABEL_ENCODER_STEP2,
    AGE_CLASS_MAP
) = assets_loaded

FEATURE_LIST_WITH_GENDER = ["gender"] + FEATURE_LIST 
# Initialize database
init_db()

# -----------------------
# Helper Functions
# -----------------------

def allowed_file(filename):
    return filename.lower().endswith(tuple(ALLOWED_EXTENSIONS))

# -----------------------
# Routes
# -----------------------

@routes.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@routes.route("/home", methods=["GET"])
def home():
    return render_template("home.html")

@routes.route("/predict", methods=["POST"])
def predict():
    api_key = request.headers.get("X-API-KEY")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, plan FROM users WHERE api_key = ?", (api_key,))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return jsonify({"error": "Invalid API key"}), 403

    user_id, plan = user
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("SELECT request_count FROM usage WHERE user_id=? AND date=?", (user_id, today))
    row = cursor.fetchone()

    if plan == "free":
        if row and row[0] >= 5:
            conn.close()
            return jsonify({"error": "Free plan limit reached (5/day)"}), 429
        elif row:
            cursor.execute("UPDATE usage SET request_count = request_count + 1 WHERE user_id=? AND date=?", (user_id, today))
        else:
            cursor.execute("INSERT INTO usage (user_id, date, request_count) VALUES (?, ?, 1)", (user_id, today))

    file = request.files.get("audio")
    if not file or not allowed_file(file.filename):
        conn.close()
        return jsonify({"error": "No valid file uploaded"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save the file
    file.save(filepath)
    
    try:
        features = extract_features(filepath)
        if features is None:
            conn.close()
            os.remove(filepath)  # Clean up file
            return jsonify({"error": "Failed to extract features"}), 500

        features_df = pd.DataFrame([features], columns=FEATURE_LIST)
        features_scaled_gender = SCALER_GENDER.transform(features_df)

        best_model, best_pred, best_conf = None, None, 0
        for name, model in GENDER_MODELS.items():
            pred = model.predict(features_scaled_gender)[0]
            conf = model.predict_proba(features_scaled_gender)[0].max() * 100
            if conf > best_conf:
                best_model, best_pred, best_conf = name, pred, conf

        gender = "Female" if best_pred == 1 else "Male"
        features = [float(best_pred)] + features  # Insert gender first

        # Create fresh DataFrame with correct feature order for age prediction
        features_df = pd.DataFrame([features], columns=FEATURE_LIST_WITH_GENDER)

        features_df = features_df[SCALER_STEP1.feature_names_in_]
        logger.info(SCALER_STEP1.feature_names_in_)
        logger.info(features_df.columns.tolist())

        features_scaled_step1 = SCALER_STEP1.transform(features_df)
        step1_pred_encoded = MODEL_STEP1.predict(features_scaled_step1)[0]
        step1_pred = LABEL_ENCODER_STEP1.inverse_transform([step1_pred_encoded])[0]

        if step1_pred == 'child':
            age_group = 'child'
            age_confidence = MODEL_STEP1.predict_proba(features_scaled_step1)[0].max() * 100
        else:
            features_scaled_step2 = SCALER_STEP2.transform(features_df)
            step2_pred_encoded = MODEL_STEP2.predict(features_scaled_step2)[0]
            mapped_label = AGE_CLASS_MAP.get(step2_pred_encoded)
            age_group = mapped_label if mapped_label else LABEL_ENCODER_STEP2.inverse_transform([step2_pred_encoded])[0]
            age_confidence = MODEL_STEP2.predict_proba(features_scaled_step2)[0].max() * 100

        logger.info(f"Prediction: {gender} ({best_conf:.2f}%), Age group: {age_group} ({age_confidence:.2f}%)")

        cursor.execute("""
            INSERT INTO predictions (
                audio_file, predicted_gender, predicted_age_group,
                confidence_score, gender_confidence, age_confidence,
                is_correct, features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filename, gender, age_group,
            age_confidence, best_conf, age_confidence,
            -1, json.dumps([float(f) for f in features])
        ))
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        if conn:
            conn.close()
        return jsonify({"error": "Internal server error"}), 500
    
    finally:
        # Always try to clean up the file, even if an error occurred
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.error(f"Failed to delete file {filepath}: {str(e)}")

    return jsonify({
        "id": prediction_id,
        "gender": gender,
        "gender_confidence": best_conf,
        "age_group": age_group,
        "age_confidence": age_confidence
    })

@routes.route("/feedback", methods=["POST"])
def feedback_submit():
    data = request.form
    prediction_id = data.get("id")
    is_correct = int(data.get("is_correct", -1))
    corrected_gender = data.get("corrected_gender")
    corrected_age_group = data.get("corrected_age_group")
    user_feedback = data.get("user_feedback")

    try:
        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE predictions SET
                is_correct = ?,
                corrected_gender = ?,
                corrected_age_group = ?,
                user_feedback = ?
            WHERE id = ?
        """, (
            is_correct, corrected_gender, corrected_age_group,
            user_feedback, prediction_id
        ))
        conn.commit()
        conn.close()
        flash("✅ Thank you for your feedback!", "success")
        return redirect("/home")
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}", exc_info=True)
        flash("❌ Failed to save feedback. Try again.", "error")
        return redirect("/home")


@routes.route("/feedback", methods=["GET"])
def feedback_form():
    prediction_id = request.args.get("id")
    return render_template("feedback.html", prediction_id=prediction_id)


@routes.route("/admin/view-feedback", methods=["GET"])
def view_feedback():
    try:
        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, audio_file, predicted_gender, predicted_age_group,
                   is_correct, corrected_gender, corrected_age_group, 
                   user_feedback, timestamp 
            FROM predictions
            WHERE is_correct != -1
            ORDER BY timestamp DESC
        """)
        feedback_data = cursor.fetchall()
        conn.close()

        html_template = """
        <h2>📋 Feedback Submissions</h2>
        <table border="1" cellpadding="8">
            <tr>
                <th>ID</th><th>Audio</th><th>Gender</th><th>Age</th><th>Correct?</th>
                <th>Corrected Gender</th><th>Corrected Age</th><th>Comment</th><th>Time</th>
            </tr>
            {% for row in data %}
            <tr>
                {% for col in row %}
                <td>{{ col }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </table>
        """
        return render_template_string(html_template, data=feedback_data)
    except Exception as e:
        logger.error(f"Error loading feedback: {str(e)}")
        return "Failed to load feedback", 500



@routes.route("/api-docs", methods=["GET"])
def api_docs():
    return render_template("api_docs.html")

@routes.route("/register", methods=["POST"])
def register_submit():
    # Check if user is logged in (session exists)
    if 'user_id' in session:
        # Generate API key for logged-in user
        api_key = secrets.token_hex(16)
        try:
            conn = sqlite3.connect("predictions.db")
            cursor = conn.cursor()
            # Update existing user with API key
            cursor.execute("UPDATE users SET api_key = ? WHERE id = ?", 
                         (api_key, session['user_id']))
            conn.commit()
            conn.close()
            return jsonify({"message": "API key generated", "api_key": api_key})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # Original registration flow for non-logged-in users
    email = request.form.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400

    api_key = secrets.token_hex(16)
    try:
        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (email, api_key) VALUES (?, ?)", (email, api_key))
        conn.commit()
        conn.close()
        return jsonify({"message": "Account created", "api_key": api_key})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered"}), 409

@routes.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")



DB_PATH = 'predictions.db'

# -----------------------
# Show login and signup pages
# -----------------------

@routes.route("/login", methods=["GET"])
def show_login():
    return render_template("login.html")

@routes.route("/sign_up", methods=["GET"])
def show_sign_up():
    return render_template("sign_up.html")


# -----------------------
# Handle Sign Up
# -----------------------



@routes.route("/google/callback")
def google_auth_callback():
    if not google.authorized:
        flash("Google authorization failed.", "error")
        return redirect("/login")

    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        flash("Failed to fetch user info from Google.", "error")
        return redirect("/login")

    user_info = resp.json()
    email = user_info.get("email")

    if not email:
        flash("No email returned by Google.", "error")
        return redirect("/login")

    # Check if user exists
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT id, email FROM users WHERE email = ?", (email,))
    user = c.fetchone()

    if not user:
        # Register new user
        try:
            c.execute("INSERT INTO users (email) VALUES (?)", (email,))
            conn.commit()
            user_id = c.lastrowid
        except IntegrityError:
            conn.close()
            flash("Error creating Google account.", "error")
            return redirect("/login")
    else:
        user_id = user[0]

    conn.close()

    # Set session
    session["user_id"] = user_id
    session["user_email"] = email
    flash(f"🔓 Signed in as {email} via Google!", "success")
    return redirect("/home")


@routes.route('/sign_up', methods=['POST'])
def sign_up():
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    # 1. Check required fields
    if not email or not password or not confirm_password:
        flash("All fields are required.", "error")
        return redirect('/sign_up')

    # 2. Check if passwords match
    if password != confirm_password:
        flash("Passwords do not match.", "error")
        return redirect('/sign_up')

    # 3. Validate password strength
    if len(password) < 8 or not re.search(r"[A-Z]", password) \
       or not re.search(r"[a-z]", password) \
       or not re.search(r"[0-9]", password) \
       or not re.search(r"[^A-Za-z0-9]", password):
        flash("Password must be at least 8 characters, contain a capital letter, a number, and a special character.", "error")
        return redirect('/sign_up')

    # 4. Check if user exists
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    existing_user = c.fetchone()

    if existing_user:
        conn.close()
        flash("Email already registered.", "error")
        return redirect('/sign_up')

    # 5. Save new user
    password_hash = generate_password_hash(password)
    try:
        c.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, password_hash))
        conn.commit()
    except Exception as e:
        flash("Something went wrong. Try again.", "error")
    finally:
        conn.close()

    flash("Account created successfully! Please log in.", "success")
    return redirect('/login')


# -----------------------
# Handle Login
# -----------------------

@routes.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')

    # 1. Validate input
    if not email or not password:
        flash("All fields are required.", "error")
        return redirect('/login')

    # 2. Check if user exists and verify password
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()

    if not user or not check_password_hash(user[2], password):
        flash("Invalid email or password.", "error")
        return redirect('/login')

    # 3. Set session
    session['user_id'] = user[0]
    session['user_email'] = user[1]

    flash(f"Welcome back, {user[1]}!", "success")
    return redirect('/home')  # Change as needed

# -----------------------
# Handle logout
# -----------------------

@routes.route("/logout", methods=["GET"])
def logout():
    return redirect("/login")  # Redirect to login page after logout