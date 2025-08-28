import streamlit as st
import numpy as np
from PIL import Image
from utils.image_processing import gaussian_blur_app

# Load the segmentation pipeline with caching
@st.cache_resource
def load_model():
    from transformers import pipeline
    return pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

def main():
    st.set_page_config(page_title="Background Blur App", layout="wide")
    
    st.title("Gaussian Blur App")
    st.markdown("Upload an image and adjust the sigma value to blur the background while keeping the foreground sharp.")
    
    # Initialize the model
    try:
        pipe = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        sigma = st.slider(
            "Blur Intensity (Sigma)",
            min_value=1.0,
            max_value=50.0,
            value=5.0,
            step=0.5
        )
        
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )
    
    # Main content area
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_column_width=True)
        
        with col2:
            st.subheader("Processed Image")
            with st.spinner("Processing image..."):
                try:
                    result_image = gaussian_blur_app(original_image, sigma, pipe)
                    st.image(result_image, use_column_width=True)
                    
                    # Download button for processed image
                    buf = io.BytesIO()
                    result_image.save(buf, format="PNG")
                    st.download_button(
                        label="Download Processed Image",
                        data=buf.getvalue(),
                        file_name="blurred_background.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error processing image: {e}")
    else:
        st.info("Please upload an image to get started")

if __name__ == "__main__":
    import io
    main()