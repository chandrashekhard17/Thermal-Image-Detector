# Library imports
import numpy as np
import streamlit as st
import cv2
import matplotlib.pyplot as plt

# Function to convert a grayscale image to thermal-like image
def convert_to_thermal(image):
    # Apply a colormap to simulate a thermal image
    thermal_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)  # You can try different colormaps like COLORMAP_HOT
    return thermal_image

# Setting Title of App
st.title("Thermal Image Converter")
st.markdown("Upload an image to convert it to a thermal image")

# Uploading the image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

submit = st.button('Convert to Thermal Image')

# On predict button click
if submit:
    if uploaded_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Check if the image is loaded properly
        if opencv_image is not None:
            st.image(opencv_image, channels="BGR", caption="Original Image", use_column_width=True)
            
            # Convert to grayscale (assuming input image is RGB)
            grayscale_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

            # Resize the image (you can adjust the size as per your requirement)
            grayscale_image = cv2.resize(grayscale_image, (256, 256))

            # Convert the grayscale image to a thermal image
            thermal_image = convert_to_thermal(grayscale_image)

            # Check if thermal image is created correctly
            if thermal_image is not None:
                st.image(thermal_image, channels="BGR", caption="Thermal Image", use_column_width=True)
            else:
                st.error("Failed to convert to thermal image.")
            
            # Optionally: Show histogram of the grayscale image
            st.subheader("Image Histogram")
            fig, ax = plt.subplots()
            ax.hist(grayscale_image.ravel(), bins=256, color='gray', alpha=0.7)
            ax.set_xlim([0, 256])
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            # Optionally: Save the thermal image
            if st.button("Save Thermal Image"):
                save_path = "thermal_image_output.png"
                cv2.imwrite(save_path, thermal_image)
                st.success(f"Thermal image saved at {save_path}")
        else:
            st.error("Failed to load image. Please try again.")
