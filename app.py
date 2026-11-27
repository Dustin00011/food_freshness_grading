import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import json
import tempfile
import zipfile

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Food Frehness",
    page_icon="🍎",
    layout="wide"
)

CLASS_NAMES = ["Fresh", "Semi-Fresh", "Rotten"]

BASE_DIR = Path(__file__).resolve().parent
CUSTOM_MODEL_PATH = BASE_DIR / "models" / "best_custom_cnn_fixed.keras"
MOBILENET_MODEL_PATH = BASE_DIR / "models" / "best_mobilenetv2.keras"


# --------------------------------
# LOAD TRAINED MODELS
# --------------------------------

def remove_unsupported_config_fields(value):
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        for nested_value in value.values():
            remove_unsupported_config_fields(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            remove_unsupported_config_fields(nested_value)


def create_compatible_model_copy(model_path):
    compatible_dir = Path(tempfile.gettempdir()) / "freshcheck_keras"
    compatible_dir.mkdir(parents=True, exist_ok=True)

    compatible_path = compatible_dir / model_path.name
    if compatible_path.exists():
        if compatible_path.stat().st_mtime >= model_path.stat().st_mtime:
            return compatible_path

    with zipfile.ZipFile(model_path, "r") as source:
        with zipfile.ZipFile(compatible_path, "w", zipfile.ZIP_DEFLATED) as destination:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "config.json":
                    config = json.loads(data.decode("utf-8"))
                    remove_unsupported_config_fields(config)
                    data = json.dumps(config, separators=(",", ":")).encode("utf-8")
                destination.writestr(item, data)

    return compatible_path


@st.cache_resource
def load_models():
    missing = [
        str(path)
        for path in (CUSTOM_MODEL_PATH, MOBILENET_MODEL_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))

    custom_compatible_path = create_compatible_model_copy(CUSTOM_MODEL_PATH)
    mobile_compatible_path = create_compatible_model_copy(MOBILENET_MODEL_PATH)

    custom_model = tf.keras.models.load_model(
        str(custom_compatible_path),
        compile=False
    )
    mobile_model = tf.keras.models.load_model(
        str(mobile_compatible_path),
        compile=False
    )
    return custom_model, mobile_model


try:
    custom_model, mobile_model = load_models()
except Exception as exc:
    st.error(f"The trained models could not be loaded: {exc}")
    st.stop()


# --------------------------------
# IMAGE PREPROCESSING
# --------------------------------

def preprocess_custom(image):
    image = image.resize((128, 128))
    image = np.array(image).astype("float32")
    image = image / 127.5 - 1.0
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_mobilenet(image):
    image = image.resize((224, 224))
    image = np.array(image).astype("float32")
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


# --------------------------------
# PREDICTION FUNCTIONS
# --------------------------------

def predict_custom(image):
    processed_image = preprocess_custom(image)
    predictions = custom_model.predict(processed_image, verbose=0)[0]
    return predictions


def predict_mobilenet(image):
    processed_image = preprocess_mobilenet(image)
    predictions = mobile_model.predict(processed_image, verbose=0)[0]
    return predictions


# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.title("🍎 FreshCheck AI")

st.subheader(
    " Food Freshness Classification System"
)

st.write(
    "Upload an image of a fruit or vegetable "
    "to analyse its freshness using two deep learning models."
)

st.divider()


# --------------------------------
# IMAGE UPLOAD
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Food Image",
        width=350
    )

    if st.button("Analyse Freshness", type="primary"):

        with st.spinner("Analysing image..."):
            custom_predictions = predict_custom(image)
            mobile_predictions = predict_mobilenet(image)

        st.success("Analysis completed!")

        # --------------------------------
        # PREDICTION RESULTS
        # --------------------------------

        st.header("Model Predictions")

        col1, col2 = st.columns(2)

        # CUSTOM CNN
        with col1:

            st.subheader("Custom CNN")

            custom_index = np.argmax(custom_predictions)

            st.metric(
                "Predicted Freshness",
                CLASS_NAMES[custom_index]
            )

            st.write("Class Confidence Scores")

            for i, class_name in enumerate(CLASS_NAMES):
                st.write(
                    f"{class_name}: "
                    f"{custom_predictions[i] * 100:.2f}%"
                )
                st.progress(float(custom_predictions[i]))

        # MOBILENETV2
        with col2:

            st.subheader("MobileNetV2")

            mobile_index = np.argmax(mobile_predictions)

            st.metric(
                "Predicted Freshness",
                CLASS_NAMES[mobile_index]
            )

            st.write("Class Confidence Scores")

            for i, class_name in enumerate(CLASS_NAMES):
                st.write(
                    f"{class_name}: "
                    f"{mobile_predictions[i] * 100:.2f}%"
                )
                st.progress(float(mobile_predictions[i]))

        # --------------------------------
        # MODEL COMPARISON
        # --------------------------------

        st.divider()

        st.header("Model Comparison")

        if custom_index == mobile_index:
            st.success(
                f"Both models predict: {CLASS_NAMES[custom_index]}"
            )
        else:
            st.warning(
                "The models produced different predictions."
            )

        comparison_col1, comparison_col2 = st.columns(2)

        with comparison_col1:
            st.write("**Custom CNN Prediction**")
            st.write(CLASS_NAMES[custom_index])

        with comparison_col2:
            st.write("**MobileNetV2 Prediction**")
            st.write(CLASS_NAMES[mobile_index])

else:

    st.info(
        "Please upload an image to begin freshness classification."
    )


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "FreshCheck AI | Custom CNN vs MobileNetV2 | "
    "Food Freshness Classification Project"
)
