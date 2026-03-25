import streamlit as st
from PIL import Image
from prediction import predict

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Mango Disease Detection",
    layout="wide"
)

# ----------------------------
# Title
# ----------------------------
st.title("🥭 Mango Disease AI Assistant")
st.write("🌿 Take a picture of your mango leaf, fruit, or stem, and let our AI help detect any diseases early!")

st.write("---")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("🥭 Follow the instructions:")
st.sidebar.write(
"""
1. 📷 Upload a mango image (leaf, fruit, or stem) (jpg, png, jpeg).  
2. 🔍 Click 'Detect Disease'.  
3. 🧪 View the predicted disease and confidence score.  
4. 🌱 Take preventive action if disease is detected.
"""
)

st.sidebar.write("---")

# ----------------------------
# File uploader
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload Mango Leaf Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ----------------------------
# Prediction section
# ----------------------------
if uploaded_files:

    st.subheader("Prediction Results")

    cols = st.columns(3)

    for idx, uploaded_file in enumerate(uploaded_files):

        img = Image.open(uploaded_file).convert("RGB")

        class_name, confidence = predict(uploaded_file)

        with cols[idx % 3]:

            st.image(img, use_container_width=True)

            st.write(f"**Disease:** {class_name}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")

else:
    st.info("Please upload one or more mango leaf images to start prediction.")

st.write("---")
