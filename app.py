"""
🌿 Smart Recycling Classifier — Streamlit Web App
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from config import *
from src.models.model import load_model
from src.models.predict import preprocess_image

# ============================================================
# 🎨 PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="♻️ Smart Recycling Classifier",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# 🎨 CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        color: #2E7D32;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-left: 6px solid #2E7D32;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #FFF8E1, #FFECB3);
        border-left: 6px solid #F9A825;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .tip-box {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-left: 6px solid #1565C0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stProgress > div > div {
        background-color: #2E7D32;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 📚 RECYCLING INFO DATABASE
# ============================================================
RECYCLING_INFO = {
    "cardboard": {
        "emoji": "📦",
        "color": "#795548",
        "bin": "♻️ Recycling Bin",
        "tips": [
            "Flatten boxes before recycling",
            "Remove tape and staples if possible",
            "Keep dry — wet cardboard cannot be recycled",
            "Pizza boxes with grease go in compost, not recycling"
        ],
        "facts": "Recycling 1 ton of cardboard saves 17 trees! 🌳"
    },
    "glass": {
        "emoji": "🍶",
        "color": "#0288D1",
        "bin": "♻️ Glass Recycling Bin",
        "tips": [
            "Rinse bottles and jars before recycling",
            "Remove metal lids separately",
            "Do NOT include broken glass — it's hazardous",
            "Separate by color if required in your area"
        ],
        "facts": "Glass can be recycled endlessly without losing quality! ✨"
    },
    "metal": {
        "emoji": "🥫",
        "color": "#546E7A",
        "bin": "♻️ Metal Recycling Bin",
        "tips": [
            "Rinse food cans thoroughly",
            "Crush cans to save space",
            "Aluminum foil can be recycled when clean",
            "Aerosol cans must be completely empty"
        ],
        "facts": "Recycling aluminum saves 95% of the energy needed to make new aluminum! ⚡"
    },
    "paper": {
        "emoji": "📄",
        "color": "#8D6E63",
        "bin": "♻️ Paper Recycling Bin",
        "tips": [
            "Keep paper clean and dry",
            "Shredded paper goes in a sealed bag",
            "Remove plastic windows from envelopes",
            "Newspaper, magazines, and office paper are all recyclable"
        ],
        "facts": "One ton of recycled paper saves 7,000 gallons of water! 💧"
    },
    "plastic": {
        "emoji": "🧴",
        "color": "#F57C00",
        "bin": "♻️ Plastic Recycling Bin",
        "tips": [
            "Check the recycling number (1-7) on the bottom",
            "Rinse containers before recycling",
            "Plastic bags go to grocery store drop-offs, NOT curbside",
            "Bottle caps can stay on in most areas"
        ],
        "facts": "Only 9% of plastic ever produced has been recycled. Every bottle counts! 🌍"
    },
    "trash": {
        "emoji": "🗑️",
        "color": "#D32F2F",
        "bin": "🗑️ General Waste Bin",
        "tips": [
            "Reduce waste by choosing reusable alternatives",
            "Check if any parts can be separated for recycling",
            "Consider composting food waste",
            "Look for special disposal programs for electronics"
        ],
        "facts": "The average person generates 4.4 lbs of trash daily. Small changes make a big difference! 💪"
    }
}


# ============================================================
# 🤖 LOAD MODEL (cached)
# ============================================================
@st.cache_resource
def get_model():
    try:
        model = load_model("recycle_net_final.keras")
        return model
    except Exception as e:
        st.error(f"❌ Model not found: {e}")
        return None


# ============================================================
# 🔮 PREDICT FUNCTION
# ============================================================
def predict(image: Image.Image):
    """Run prediction on a PIL image"""
    model = get_model()
    if model is None:
        return None, None

    # Save temp & preprocess
    temp_path = "/tmp/upload_temp.jpg"
    image.save(temp_path)

    img_array = preprocess_image(temp_path)
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]

    predicted_idx = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx]

    # Build top-3
    top3_idx = np.argsort(probabilities)[::-1][:3]
    top3 = [(CLASS_NAMES[i], float(probabilities[i])) for i in top3_idx]

    return predicted_class, confidence, top3


# ============================================================
# 🖥️ SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ♻️ About")
    st.info(
        "This AI model classifies waste into **6 categories** "
        "to help you recycle correctly!\n\n"
        "**Categories:**\n"
        "- 📦 Cardboard\n"
        "- 🍶 Glass\n"
        "- 🥫 Metal\n"
        "- 📄 Paper\n"
        "- 🧴 Plastic\n"
        "- 🗑️ Trash"
    )
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Architecture**: MobileNetV2
    - **Classes**: 6
    - **Input Size**: 224×224
    - **Training**: Transfer Learning
    """)
    st.markdown("---")
    st.caption("🌿 Built with TensorFlow & Streamlit")


# ============================================================
# 🏠 MAIN PAGE
# ============================================================
st.markdown('<p class="main-title">♻️ Smart Recycling Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image of waste and AI will tell you how to recycle it!</p>', unsafe_allow_html=True)

st.markdown("---")

# ── Upload ────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📸 Upload a waste image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)

# ── Camera Input ──────────────────────────────────────────
st.markdown("**or use your camera:**")
camera_image = st.camera_input("📷 Take a photo")

# ── Select source ─────────────────────────────────────────
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

# ============================================================
# 🔮 PREDICTION & RESULTS
# ============================================================
if image is not None:

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🖼️ Your Image")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### 🔮 Analyzing...")
        with st.spinner("🤖 AI is thinking..."):
            result = predict(image)

        if result[0] is not None:
            predicted_class, confidence, top3 = result
            info = RECYCLING_INFO[predicted_class]

            # ── Main result ───────────────────────────────
            st.markdown(f"""
            <div class="result-box">
                <h2 style="margin:0; color:#2E7D32;">{info['emoji']} {predicted_class.upper()}</h2>
                <p style="margin:5px 0; font-size:1.2rem;">🎯 Confidence: <strong>{confidence*100:.1f}%</strong></p>
                <p style="margin:5px 0;">🗑️ Bin: <strong>{info['bin']}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence bar ────────────────────────────
            st.markdown("#### 📊 Top Predictions")
            for cls, prob in top3:
                emoji = RECYCLING_INFO[cls]["emoji"]
                st.markdown(f"**{emoji} {cls.capitalize()}**")
                st.progress(float(prob))
                st.caption(f"{prob*100:.1f}%")

    st.markdown("---")

    # ── Recycling Tips ────────────────────────────────────
    if result[0] is not None:
        info = RECYCLING_INFO[predicted_class]

        st.markdown(f"### {info['emoji']} Recycling Tips for {predicted_class.capitalize()}")

        col3, col4 = st.columns([1, 1])

        with col3:
            st.markdown('<div class="tip-box">', unsafe_allow_html=True)
            st.markdown("**♻️ How to Recycle:**")
            for tip in info["tips"]:
                st.markdown(f"- ✅ {tip}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="tip-box">', unsafe_allow_html=True)
            st.markdown("**🌍 Did You Know?**")
            st.info(info["facts"])
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Low confidence warning ────────────────────────
        if confidence < 0.6:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Low Confidence!</strong> The model is not very sure about this prediction.
                Try uploading a clearer image with better lighting.
            </div>
            """, unsafe_allow_html=True)

# ── Empty state ───────────────────────────────────────────
else:
    st.markdown("""
    <div class="tip-box">
        <h4>👆 How to use:</h4>
        <ol>
            <li>Upload an image of your waste item above</li>
            <li>Or take a live photo using your camera</li>
            <li>AI will classify it into one of 6 categories</li>
            <li>Follow the recycling tips provided!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 Example Items to Try")
    cols = st.columns(3)
    examples = [
        ("📦", "Cardboard Box"),
        ("🍶", "Glass Bottle"),
        ("🥫", "Metal Can"),
        ("📄", "Newspaper"),
        ("🧴", "Plastic Bottle"),
        ("🗑️", "General Trash"),
    ]
    for i, (emoji, name) in enumerate(examples):
        with cols[i % 3]:
            st.markdown(f"**{emoji} {name}**")

