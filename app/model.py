import os
import joblib
import logging

# ============================
# Logging Setup
# ============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================
# Paths
# ============================
MODEL_DIR = os.path.join(os.getcwd(), "models2")
GENDER_MODELS_PATHS = {
    "svm": os.path.join(MODEL_DIR, "gender_model_svm.pkl"),
    "lr": os.path.join(MODEL_DIR, "gender_model_lr.pkl"),
}
SCALER_GENDER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.pkl")

STEP1_MODEL_PATH = os.path.join(MODEL_DIR, "model_step1.joblib")
STEP1_SCALER_PATH = os.path.join(MODEL_DIR, "scaler_step1.joblib")
STEP1_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_step1.joblib")

STEP2_MODEL_PATH = os.path.join(MODEL_DIR, "model_step2.joblib")
STEP2_SCALER_PATH = os.path.join(MODEL_DIR, "scaler_step2.joblib")
STEP2_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_step2.joblib")

AGE_CLASS_MAP = {
    0: 'eighties',
    1: 'fifties',
    2: 'fourties',
    3: 'seventies',
    4: 'sixties',
    5: 'teen',
    6: 'thirties',
    7: 'twenties'
}

_cached_assets = None

# ============================
# Model Loader
# ============================
def load_assets():
    global _cached_assets
    if _cached_assets is not None:
        return _cached_assets

    try:
        logger.info("🔄 Loading models and scalers...")

        gender_models = {
            name: joblib.load(path) for name, path in GENDER_MODELS_PATHS.items()
        }
        scaler_gender = joblib.load(SCALER_GENDER_PATH)
        feature_list = joblib.load(FEATURE_LIST_PATH)

        model_step1 = joblib.load(STEP1_MODEL_PATH)
        scaler_step1 = joblib.load(STEP1_SCALER_PATH)
        encoder_step1 = joblib.load(STEP1_ENCODER_PATH)

        model_step2 = joblib.load(STEP2_MODEL_PATH)
        scaler_step2 = joblib.load(STEP2_SCALER_PATH)
        encoder_step2 = joblib.load(STEP2_ENCODER_PATH)

        logger.info("✅ All assets loaded successfully.")

        _cached_assets = (
            gender_models, scaler_gender, feature_list,
            model_step1, scaler_step1, encoder_step1,
            model_step2, scaler_step2, encoder_step2,
            AGE_CLASS_MAP
        )
        return _cached_assets

    except Exception as e:
        logger.error(f"❌ Error loading assets: {e}")
        raise
