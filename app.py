import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

# ================= CUSTOM CSS =================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

/* Header */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: white;
    margin-top: 10px;
}

.sub-title {
    text-align: center;
    font-size: 22px;
    color: #cbd5e1;
    margin-bottom: 40px;
}

/* Section Box */
.section-box {
    background-color: #111827;
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0px 0px 12px rgba(255,255,255,0.1);
}

/* Prediction Box */
.prediction-box {
    background-color: #0f766e;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

/* Recommendation Box */
.recommend-box {
    background-color: #7f1d1d;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}

/* Text */
.section-title {
    font-size: 28px;
    font-weight: bold;
    color: #38bdf8;
    margin-bottom: 15px;
}

.normal-text {
    font-size: 18px;
    color: #e2e8f0;
    line-height: 1.8;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================

st.markdown(
    """
    <div class='main-title'>
    🚧 AI-Based Road Damage Detection System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='sub-title'>
    Smart City Infrastructure Monitoring using CNN
    </div>
    """,
    unsafe_allow_html=True
)

# ================= ABOUT PROJECT =================

st.markdown("<div class='section-box'>", unsafe_allow_html=True)

st.markdown(
    "<div class='section-title'>📌 About the Project</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='normal-text'>

    • Road monitoring is important to prevent accidents and improve public safety.<br><br>

    • Damaged roads like potholes and cracks can increase vehicle damage and traffic risks.<br><br>

    • Convolutional Neural Networks (CNNs) help computers analyze and classify road surface images automatically.<br><br>

    • AI-based monitoring systems are widely used in smart cities, transportation systems, autonomous vehicles, and infrastructure maintenance.

    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)

# ================= MODEL =================

# Replace this with your trained model later
# Example:
# model = joblib.load("road_damage_model.pkl")

classes = [
    "Crack",
    "Manhole",
    "Pothole"
]

# ================= IMAGE UPLOAD =================

st.markdown("<div class='section-box'>", unsafe_allow_html=True)

st.markdown(
    "<div class='section-title'>📤 Upload Road Image</div>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ================= IMAGE PREVIEW =================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>🖼 Uploaded Image Preview</div>",
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= FAKE PREDICTION =================
    # Replace with actual CNN prediction later

    probabilities = np.random.dirichlet(np.ones(3), size=1)[0]

    pred_index = np.argmax(probabilities)

    pred_label = classes[pred_index]

    confidence = probabilities[pred_index] * 100

    # ================= SEVERITY =================

    if confidence > 85:
        severity = "High"

    elif confidence > 65:
        severity = "Medium"

    else:
        severity = "Low"

    # ================= PREDICTION AREA =================

    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <h2>Prediction: {pred_label} Detected</h2>
        <h3>Confidence: {confidence:.2f}%</h3>
        <h3>Severity: {severity}</h3>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= VISUALIZATION =================

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>📊 Class Confidence Graph</div>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(7,4))

    ax.bar(classes, probabilities)

    ax.set_ylabel("Confidence")

    ax.set_ylim([0,1])

    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= RECOMMENDATIONS =================

    st.markdown("<div class='recommend-box'>", unsafe_allow_html=True)

    st.markdown(
        "<h2>⚠ Recommendations</h2>",
        unsafe_allow_html=True
    )

    if severity == "High":

        st.markdown("""
        - Immediate maintenance recommended.<br>
        - High-risk road condition detected.<br>
        - Authorities should prioritize repair work.
        """, unsafe_allow_html=True)

    elif severity == "Medium":

        st.markdown("""
        - Schedule maintenance soon.<br>
        - Moderate road damage detected.<br>
        - Regular monitoring recommended.
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        - Minor road issue detected.<br>
        - Continue periodic inspections.<br>
        - Low immediate safety risk.
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)