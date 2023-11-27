import streamlit as st
import base64
from PIL import Image
from embeddings import search



def display_images(image_paths):
    # Display images in a grid format
    for path in image_paths:
        image = Image.open(path)
        st.image(image, caption="Processed Image", use_column_width=True)

def main():
    st.title("Image Processing with Streamlit")

    # Image uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded image to base64
        image_base64 = base64.b64encode(uploaded_file.read()).decode("utf-8")

        # Process the image
        found_img_path = search(image_base64)

        # Display the processed image
        st.subheader("Processed Image")
        display_images(found_img_path)

if __name__ == "__main__":
    main()
