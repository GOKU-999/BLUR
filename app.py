import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from transformers import pipeline
import io

# Load the segmentation pipeline with caching
@st.cache_resource
def load_model():
    return pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

def gaussian_blur_app(image, sigma, pipe):
    """
    Apply Gaussian blur to the background of an image based on segmentation.
    
    Args:
        image: PIL Image uploaded by the user
        sigma: Blur intensity (sigma value for Gaussian blur)
        pipe: The segmentation pipeline
        
    Returns:
        PIL Image with blurred background
    """
    # Generate the binary mask using the pipeline
    pillow_mask = pipe(image, return_mask=True)
    
    original_image_np = np.array(image)
    mask = np.array(pillow_mask)
    
    # Apply Gaussian blur using PIL instead of OpenCV
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    blurred_image_np = np.array(blurred_image)
    
    # Convert 2D mask to 3D if necessary
    if len(mask.shape) == 2:
        mask_3d = np.stack([mask] * 3, axis=-1)
    else:
        mask_3d = mask
    
    mask_normalized = mask_3d / 255.0
    
    # Combine foreground and background
    final_image = (mask_normalized * original_image_np + (1 - mask_normalized) * blurred_image_np).astype(np.uint8)
    
    return Image.fromarray(final_image)

def main():
    st.set_page_config(page_title="Background Blur App", layout="wide")
    
    st.title("🎨 Background Blur App")
    st.markdown("Upload an image and adjust the blur intensity to create a professional-looking background blur effect.")
    
    # Initialize the model
    try:
        pipe = load_model()
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure all dependencies are installed correctly.")
        return
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Settings")
        sigma = st.slider(
            "Blur Intensity",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
            help="Higher values create more blur effect"
        )
        
        uploaded_file = st.file_uploader(
            "📁 Upload Image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, JPEG, PNG, WEBP"
        )
    
    # Main content area
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_column_width=True)
            st.caption(f"Original size: {original_image.size[0]}x{original_image.size[1]}")
        
        with col2:
            st.subheader("✨ Processed Image")
            with st.spinner("🔍 Processing image... This may take a few seconds"):
                try:
                    result_image = gaussian_blur_app(original_image, sigma, pipe)
                    st.image(result_image, use_column_width=True)
                    st.caption("Background blurred successfully!")
                    
                    # Download button for processed image
                    buf = io.BytesIO()
                    result_image.save(buf, format="PNG")
                    st.download_button(
                        label="💾 Download Processed Image",
                        data=buf.getvalue(),
                        file_name="blurred_background.png",
                        mime="image/png",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"❌ Error processing image: {e}")
                    st.info("Please try with a different image or lower blur intensity.")
    else:
        st.info("👆 Please upload an image to get started!")
        st.image("https://via.placeholder.com/600x400/3B82F6/FFFFFF?text=Upload+an+Image", use_column_width=True)

if __name__ == "__main__":
    main()
