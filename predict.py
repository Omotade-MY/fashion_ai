import tensorflow as tf
from tensorflow.keras import backend as K
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
parts = ['ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm', 'height', 'hip', 'leg-length', 'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist']
   
def annotate_image_vertically(image_file, values, output_path=None, font_scale=0.8, color=(0, 255, 20), thickness=2, start_position=(10, 30), line_spacing=30):
    """
    Annotates an image with a list of values stacked vertically and saves the annotated image.

    :param image_path: Path to the input image file.
    :param values: List of values to annotate on the image.
    :param output_path: Path to save the annotated image. If None, saves as 'annotated_<original_image_name>'.
    :param font_scale: Font scale for the text annotation.
    :param color: Color of the text (in BGR format).
    :param thickness: Thickness of the text.
    :param start_position: Starting position (x, y) for the first annotation.
    :param line_spacing: Space between lines of text.
    :return: Path to the saved annotated image.
    """
    # Read the image
    #parts = ['ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm', 'height', 'hip', 'leg-length', 'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist']
    measurements = dict(zip(parts,values))
    image =  Image.open(image_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_file}")

    # Set font for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Annotate the image with the values
    x, y = start_position
    for i, part in enumerate(parts):
        value = round(measurements[part], 1)
        text = f"{part}: {value} cm"
        # Annotate vertically by keeping x constant and increasing y
        
        cv2.putText(image, text, (x, y + i * line_spacing), font, font_scale, color, thickness)

    # Determine the output path if not provided

    # Save the annotated image
    cv2.imwrite(output_path, image)

    return output_path, image

def r2_score_k(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - ss_res/ (ss_tot + K.epsilon())

    return r2


def load_image(image_file, img_size=(224, 224)):
    try:
        image_bytes = image_bytes = image_file.read()
        image = tf.image.decode_image(image_bytes, channels=1)  # Assuming RGB images
        image = tf.image.resize(image, img_size)
        
        #img = img.resize(img_size)
        image = image / 255.0 
        image = tf.image.grayscale_to_rgb(image)
        
        #img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimensio
        return image
    except tf.errors.NotFoundError:
        print(f"Skipping Image {image_file}")


def process_input_img(img_mask_front, img_mask_left, meta):
    img_mask = tf.expand_dims(img_mask_front, axis=0)
    img_left = tf.expand_dims(img_mask_left, axis=0)
    meta = tf.expand_dims(meta, axis=0)
    return img_mask, img_left, meta


def load_model(path, custom_objects=None):
    if custom_objects is None:
        custom_objects={'r2_score_k':r2_score_k}
    model = tf.keras.models.load_model(path, custom_objects=custom_objects)
    return model

def predict_measurements(model, data):
    
    measurements = model.predict(data)

    return measurements

# Streamlit UI
st.title("Fashion AI - Body Measurement Predictor")

st.write("Upload front view and side view images, and select gender.")

# Upload images
front_image = st.file_uploader("Upload Front View Image", type=["png", "jpg", "jpeg"])
side_image = st.file_uploader("Upload Side View Image", type=["png", "jpg", "jpeg"])

# Select gender
gender = st.selectbox("Select Gender", ['Not Selected',"Male", "Female"])
if gender == 'Not Selected':
    gender = None
gender_value =  0 if gender == 'Male' else 1

if not st.session_state.get('model'):
    model = load_model("models/modelV4_20-epochs.h5")
    st.session_state['model'] = model

if front_image and side_image and gender:
  
    img_mask, img_left = load_image(front_image), load_image(side_image)
    meta = tf.convert_to_tensor([gender_value], dtype=tf.float32)
    img_mask, img_left, meta = process_input_img(img_mask_front=img_mask, img_mask_left=img_left, meta=meta)
    model = st.session_state.get('model')
    
    measurements = predict_measurements(model, data=[img_mask, img_left, meta])
    # List of values to annotate
    output_path = f"predictions_image_UI_{datetime.now().strftime('%y_%m_%d_%H_%M_%S')}.jpg" 
    # Annotate the image and save it
    annotated_image_path, annotated_image = annotate_image_vertically(front_image, measurements[0], output_path)
    annotated_image_pil = Image.fromarray(annotated_image)

    st.image(annotated_image_pil, caption="Front View Image with Body Measurements", use_column_width=True)
    for p,m in zip(parts,measurements[0]):
        st.write(f"{p}: {m} cm ")
    
