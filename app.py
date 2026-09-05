import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================
# 1. Page Configuration
# ============================================

st.set_page_config(
    page_title="Detect Covid-19",
    page_icon="⚕️",
    layout="centered"
)


# ============================================
# 2. Load the trained model
# ============================================

@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("model.keras")
    return model


model = load_my_model()


# ============================================
# 3. Class names
# ============================================

class_names = [
    "COVID",
    "Normal",
    "Lung_Opacity",
    "Viral Pneumonia",
]


# ============================================
# 4. App Title
# ============================================

st.title("⚕️ Detect Covid-19 from Chest X-ray")
st.write(
    "Upload a chest X-ray image and the trained deep learning model "
    "will predict whether the patient has Covid-19."
)


# ============================================
# 5. Upload image
# ============================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================
# 6. Process the uploaded image
# ============================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # ========================================
    # 7. Resize image
    # ========================================

    image = image.resize((224, 224))

    # ========================================
    # 8. Convert image to NumPy array
    # ========================================

    image_array = np.array(image)

    # ========================================
    # 9. Normalize image
    # ========================================

    image_array = image_array.astype("float32") / 255.0

    # ========================================
    # 10. Add batch dimension
    # ========================================

    image_array = np.expand_dims(image_array, axis=0)

    # ========================================
    # 11. Make prediction
    # ========================================

    if st.button("🔍 Predict"):

        predictions = model.predict(image_array)

        # Get predicted class
        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        # Get confidence
        confidence = predictions[0][predicted_index] * 100

        # ====================================
        # 12. Display result
        # ====================================

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        # ====================================
        # 13. Display probabilities
        # ====================================

        st.subheader("Class Probabilities")

        for i in range(len(class_names)):

            probability = predictions[0][i] * 100

            st.write(
                f"{class_names[i]}: {probability:.2f}%"
            )