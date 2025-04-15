import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile

class Cartoonizer:
    def __init__(self):
        pass

    def enhance_color(self, img):
        # Convert to HSV, increase saturation for vibrant colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 60)  # Increase saturation significantly for a more vibrant look
        enhanced_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    def sharpen_image(self, img):
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    def render(self, img):
        # Convert PIL Image to OpenCV format (BGR)
        img_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Enhance colors first
        img_rgb = self.enhance_color(img_rgb)
        
        # Resize image for better quality
        img_rgb = cv2.resize(img_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Copy image for processing
        img_color = img_rgb.copy()

        # Apply bilateral filter (less smoothing for clearer details)
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=150, sigmaSpace=150)

        # Convert to gray for edge detection
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.medianBlur(img_gray, 5)

        # Edge detection with higher threshold to make edges more visible
        img_edge = cv2.adaptiveThreshold(
            img_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 9, 5
        )

        # Convert edges back to BGR
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)

        # Sharpen the image to make colors and features more distinct
        img_color = self.sharpen_image(img_color)

        # Combine the color image and edge image for the cartoon effect
        cartoon = cv2.addWeighted(img_color, 0.7, img_edge, 0.3, 0)  # Stronger edge blending

        # Convert back to RGB format for PIL
        cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
        return cartoon_rgb

# Streamlit UI
st.title("🎨 Cartoonify Your Image")

uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    cartoonizer = Cartoonizer()
    cartoon_img = cartoonizer.render(image)

    st.image(cartoon_img, caption="Cartoonized Image", use_container_width=True)

    # Save cartoon image to temp file for download
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        save_path = f.name
        cartoon_bgr = cv2.cvtColor(cartoon_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, cartoon_bgr)

        with open(save_path, "rb") as file:
            st.download_button(
                label="💾 Download Cartoon Image",
                data=file,
                file_name="cartoon_image.jpg",
                mime="image/jpeg"
            )
