import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.optimizers.schedules import ExponentialDecay
import base64 


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_path = "../static/background.jpg"  # Replace with your image filename or full path
base64_image = get_base64_of_bin_file(img_path)

st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        background: url(data:image/jpeg;base64,{base64_image}) no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def model_prediction(test_image):
    tf.keras.utils.get_custom_objects()['ExponentialDecay'] = ExponentialDecay
    model = tf.keras.models.load_model("../models/plant_disease.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("🌿Plant Disease Detection System for Sustainable Agriculture")
st.sidebar.markdown(
    "<p style='font-size:14px;'>Empowering sustainable agriculture with AI.</p>", 
    unsafe_allow_html=True
)

app_mode = st.sidebar.radio("Navigate", ["🏠 Home", "🔬 Disease Recognition"], index=0)
import streamlit as st
from PIL import Image

if app_mode == "🏠 Home":
    # Title
    st.markdown(
        """<h1 style='text-align: center; color: green;'>🌱 Plant Disease Detection System</h1>""",
        unsafe_allow_html=True
    )
    
    # Subtitle
    st.markdown(
        """<p style='text-align: center; font-size:18px; color: blue;'>Use AI to identify plant diseases and improve agricultural practices.</p>""",
        unsafe_allow_html=True
    )
    
    # Description with Colors
    st.markdown(
    """
    <div style='font-size:16px;'>
        <p style='color: black;'><b>About the System:<br></b> The <span style='color: #951f58;'><b>Plant Disease Detection System</b></span> uses advanced machine learning algorithms to analyze images of plants and detect common diseases. By identifying diseases early, this tool helps farmers and gardeners take proactive measures to improve crop health and increase yield.</p>
        <p style='color: black;'><b>How It Works:</b></p>
        <ol>
            <li style='color: black;'>Upload an image of a plant leaf.</li>
            <li style='color: black;'>The system processes the image using AI models trained on plant disease datasets.</li>
            <li style='color: black;'>Get instant results with disease diagnosis and treatment recommendations.</li>
            <li style='color: black;'>Efficiency: Rapid and accurate disease detection.</li>
            <li style='color: black;'>Sustainability: Helps minimize the overuse of pesticides.</li>
            <li style='color: black;'>Accessibility: Simple and easy-to-use interface.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

    
    # Image paths and captions
    image_paths = ["../static/pic1.jpg", "../static/pic2.jpg", "../static/pic3.jpg", "../static/pic4.jpg", "../static/pic5.jpg", "../static/pic6.png", "../static/pic7.jpeg", "../static/pic8.jpg"]
    
    # Image Grid
    num_columns = 3
    cols = st.columns(num_columns)
    
    for idx, image_path in enumerate(image_paths):
        try:
            img = Image.open(image_path)
            with cols[idx % num_columns]:
                st.image(img, use_container_width=True)
        except FileNotFoundError:
            st.error(f"Image not found: {image_path}")

elif app_mode == "🔬 Disease Recognition":
    
    st.markdown(
        """<h1 style='text-align: center; color: green;'>🔬 <u>Disease Recognition</u></h1>""", 
        
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-size:16px;color:blue'>Upload an image of a plant leaf, and our AI model will identify the disease.</p>", 
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            "<p style='color: #8A2BE2; font-weight: bold;'>Choose a Leaf Image:</p>",
            unsafe_allow_html=True
        )
        test_image = st.file_uploader(" ", type=["jpg", "jpeg", "png"])


        if test_image is not None:
            if 'show_image' not in st.session_state:
                st.session_state.show_image = False

            # Styled toggle button
            if st.button("👁️ Show/Hide Image", use_container_width=True):
                st.session_state.show_image = not st.session_state.show_image

            # Display the uploaded image if toggled on
            if st.session_state.show_image:
                st.markdown("<p style='color: #DC143C; font-weight: bold;'>Uploaded Image:</p>", unsafe_allow_html=True)
                st.image(test_image, caption="Uploaded Image", use_container_width=True)

    with col2:  # Second column (for prediction and show button)
        if test_image is not None:
            st.markdown(
                "<p style='text-align: center;'>Image uploaded successfully! Ready to predict.</p>", 
                unsafe_allow_html=True)
            # Add Predict Disease button
            if st.button("🔍 Predict Disease"):
                with st.spinner("Analyzing the image..."):
                    result_index = model_prediction(test_image)
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                st.balloons()
                st.markdown(
                    f"""
                    <div style='border: 2px solid black; border-radius: 10px; padding: 10px; background-color: #f9f9f9;'>
                    <p style='text-align: center; font-size: 18px; color: black; margin: 0;'>
                    🌟 <b>Model Prediction:</b> It's a <b>{class_name[result_index]}</b>
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.sidebar.markdown("""
---
👨‍💻 Developed by Aditya
""", unsafe_allow_html=True)
