import streamlit as st
import tensorflow as tf
import numpy as np
import base64

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Smart Crop Disease Detection", page_icon="🌿", layout="wide")


# ---- CUSTOM BACKGROUND CSS ----
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    bg_css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main-title {{
        font-size: 42px;
        color: #ffffff;
        font-weight: bold;
        text-shadow: 2px 2px 4px black;
        text-align: center;
    }}
    .content-box {{
        background: rgba(0, 0, 0, 0.60);
        padding: 25px;
        border-radius: 12px;
        color: white;
        font-size: 18px;
        margin-top: 20px;
    }}
    h3, h4, h2 {{
        color: #fadb3a !important;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)


# Add background image here
add_bg_from_local("home_page.jpeg")     # background image



# ---- MODEL PREDICTION ----
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)



# ---- SIDEBAR ----
st.sidebar.title("📌 Dashboard")
app_mode = st.sidebar.radio("Select Page", ["Home", "About", "Disease Recognition"])



# ───────────────────────── HOME PAGE ─────────────────────────
if app_mode == "Home":
    st.markdown("<h1 class='main-title'>🌿 Smart Crop Disease Detection System</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='content-box'>
    Welcome to the <b>AI-powered Plant Disease Recognition System</b>.  
    Upload a plant leaf image and the system will instantly identify the disease and provide guidance for prevention.
    <br><br>
    🌱 Using AI for modern agriculture to help farmers reduce crop loss and improve productivity.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔍 How It Works")
    st.markdown("""
    ✔ Upload an image of a leaf  
    ✔ AI model analyzes disease patterns  
    ✔ Get name of disease + preventive suggestions
    """)

    st.markdown("---")
    st.subheader("🌟 Why This System?")
    st.markdown("""
    - Highly accurate Deep Learning model (CNN)  
    - Fast — Results in seconds  
    - Farmer-friendly and easy to use  
    - Supports smart and sustainable agriculture
    """)



# ───────────────────────── ABOUT PAGE ─────────────────────────
elif app_mode == "About":
    st.markdown("<h1 class='main-title'>📘 About This Project</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='content-box'>
    This project applies <b>Machine Learning, Deep Learning and Image Processing</b>  
    to detect diseases from tomato plant leaf images.
    <br><br>
    🧠 The model is trained on the <b>PlantVillage Dataset</b> with thousands of images of healthy and diseased tomato leaves.
    <br><br>
    🔧 <b>Technologies Used:</b> TensorFlow, Keras, OpenCV, Streamlit
    <br><br>
    🎯 <b>Goal:</b> Support farmers with fast and affordable disease diagnosis using AI.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 🔮 Future Enhancements
    🌤 Real-time drone-based monitoring  
    📱 Mobile app for instant field scanning  
    🌾 Multi-crop disease detection  
    🤖 IoT device integration for complete smart farming
    ---
    """)



# ───────────────────── DISEASE RECOGNITION PAGE ─────────────────────
elif app_mode == "Disease Recognition":
    st.markdown("<h1 class='main-title'>🩻 Disease Recognition</h1>", unsafe_allow_html=True)

    st.markdown("<div class='content-box'>Upload a tomato leaf image and let AI detect the disease.</div>",
                unsafe_allow_html=True)

    test_image = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, caption="📌 Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        if test_image is None:
            st.warning("⚠ Please upload an image first.")
        else:
            with st.spinner("⏳ Analyzing image..."):
                result_index = model_prediction(test_image)

            class_list = [
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            st.success(f"🧾 Prediction: **{class_list[result_index]}**")
            st.balloons()
