import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Chest X-Ray Detector",page_icon="⚕️",layout="wide")

# CUSTOM CSS
st.markdown(
    """
    <style>

    .main {
        padding-top: 0.5rem;
    }

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 15px;
        color: #666666;
        margin-bottom: 12px;
    }

    /* Section title */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    /* Result card */
    .result-card {
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 10px;
        text-align: center;
    }

    .prediction-title {
        font-size: 18px;
        color: #555555;
        margin-bottom: 5px;
    }

    .prediction-value {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .confidence {
        font-size: 18px;
        color: #555555;
    }

    /* Image card */
    .image-card {
        padding: 15px;
        border-radius: 15px;
        background-color: #f8f9fa;
        border: 1px solid #dddddd;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777777;
        font-size: 13px;
        margin-top: 40px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("model.keras")

    return model

model = load_my_model()
class_names = ["COVID","Normal","Lung Opacity","Viral Pneumonia"]

# SIDEBAR : 

with st.sidebar:

    st.header("⚕️ About the Model")

    st.write(
        "This application uses a pretrained "
        "MobileNetV2 deep learning model "
        "for chest X-ray classification."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Architecture:** MobileNetV2")
    st.write("**Input Size:** 224 × 224")
    st.write("**Classes:** 4")

    st.divider()

    st.subheader("Classes")

    st.write("1- COVID")
    st.write("2- Normal")
    st.write("3- Lung Opacity")
    st.write("4- Viral Pneumonia")

#  MAIN HEADER : 
st.markdown(
    '<div class="main-title">⚕️ Chest X-Ray Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-powered chest X-ray classification using MobileNetV2
    </div>
    """,
    unsafe_allow_html=True
)

# INTRODUCTION : 

st.info(
    "Upload a chest X-ray image and the trained deep learning "
    "model will classify it into one of four categories."
)



# IMAGE UPLOAD : 

st.markdown('<div class="section-title">📤 Upload Chest X-Ray : </div>',unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a chest X-ray image",type=["jpg", "jpeg", "png"],help="Supported formats: JPG, JPEG, PNG")

# PROCESS IMAGE : 

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    predict_button = st.button(
        "🔍 Predict X-Ray",
        width=450
    )
 
    if not predict_button:

        col1, col2 = st.columns(
            [1.2, 0.8]
        )

        with col1:

            st.markdown(
                '<div class="section-title">🩻 Uploaded X-Ray</div>',
                unsafe_allow_html=True
            )

            st.image(
                image,
                caption="Uploaded Chest X-Ray",
                width=450
            )

        with col2:

            st.markdown(
                '<div class="section-title">ℹ️ Image Information</div>',
                unsafe_allow_html=True
            )

            st.write(
                f"**Original image size:** "
                f"{image.size[0]} × {image.size[1]} pixels"
            )

            st.write(
                "**Ready for prediction.**"
            )

    if predict_button:

        with st.spinner("Analyzing X-ray..."):

            image_resized = image.resize(
                (224, 224)
            )

            image_array = np.array(
                image_resized
            )

            image_array = (
                image_array.astype("float32") / 255.0
            )

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = np.argmax(
                predictions[0]
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = (
                predictions[0][predicted_index] * 100
            )

        # RESULT SECTION
        
        st.write(
            f"### 📊 Prediction Image : {predicted_class}"
        )

        col1, col2 = st.columns(
            [1.1, 1]
        )


        with col1:

            st.markdown(
                '<div class="section-title">🩻 X-Ray Image :</div>',
                unsafe_allow_html=True
            )

            st.image(
                image,
                caption="Uploaded Chest X-Ray",
                width=450
            )

        with col2:

            st.write("### Confidence :")

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                float(confidence / 100)
            )


            st.write("### 📈 Class Probabilities :")

            for i in range(
                len(class_names)
            ):

                probability = (
                    predictions[0][i] * 100
                )

                st.write(
                    f"**{class_names[i]}:** "
                    f"{probability:.2f}%"
                )

                st.progress(
                    float(probability / 100)
                )

st.divider()

# FOOTER
st.markdown(
    """
    <div class="footer">
    Developed using TensorFlow, Keras, MobileNetV2 and Streamlit.
    </div>
    """,
    unsafe_allow_html=True
)