import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Initialize MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Load glasses image
glasses_image = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)

def overlay_glasses(image, face_landmarks, faceglasses_image):
    # Get the eye positions
    left_eye = face_landmarks[33]  # Left eye center
    right_eye = face_landmarks[133]  # Right eye center

    # Access x and y attributes correctly
    left_eye_x, left_eye_y = left_eye.x, left_eye.y
    right_eye_x, right_eye_y = right_eye.x, right_eye.y

    # Calculate the position and size for the glasses
    eye_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
    glasses_width = int(np.linalg.norm(np.array([left_eye_x, left_eye_y]) - np.array([right_eye_x, right_eye_y])) * 1.5)
    glasses_height = int(glasses_width * faceglasses_image.shape[0] / faceglasses_image.shape[1])

    # Debugging output
    print(f"Eye Center: {eye_center}, Glasses Width: {glasses_width}, Glasses Height: {glasses_height}")

    # Ensure dimensions are valid
    if glasses_width <= 0 or glasses_height <= 0:
        raise ValueError("Calculated glasses dimensions are invalid. Width and height must be greater than zero.")

    # Resize faceglasses image
    faceglasses_resized = cv2.resize(faceglasses_image, (glasses_width, glasses_height))

    # Calculate the position to overlay the glasses
    x_offset = eye_center[0] - glasses_width // 2
    y_offset = eye_center[1] - glasses_height // 2

    # Check if the faceglasses image has an alpha channel
    if faceglasses_resized.shape[2] == 4:  # If it has an alpha channel
        for c in range(0, 3):
            image[y_offset:y_offset + glasses_height, x_offset:x_offset + glasses_width, c] = \
                faceglasses_resized[:, :, c] * (faceglasses_resized[:, :, 3] / 255.0) + \
                image[y_offset:y_offset + glasses_height, x_offset:x_offset + glasses_width, c] * (1.0 - faceglasses_resized[:, :, 3] / 255.0)
    else:  # If it does not have an alpha channel
        image[y_offset:y_offset + glasses_height, x_offset:x_offset + glasses_width] = faceglasses_resized

    return image
def main():
    st.title("Virtual Try-On Application")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Initialize face mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(image_rgb)

            image_with_glasses = image_rgb.copy()  # Initialize the variable

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    try:
                        # Draw landmarks
                        mp_drawing.draw_landmarks(image_with_glasses, face_landmarks, mp_face_mesh.FACEMESH_IRISES)
                        # Pass face_landmarks.landmark to overlay_glasses
                        image_with_glasses = overlay_glasses(image_with_glasses, face_landmarks.landmark, faceglasses_image)
                    except ValueError as e:
                        print(f"Error drawing landmarks: {e}")
                        continue  # Skip to the next face if there's an error

        # Convert back to RGB for display
        image_with_glasses = cv2.cvtColor(image_with_glasses, cv2.COLOR_BGR2RGB)
        st.image(image_with_glasses, caption='Image with Faceglasses', use_column_width=True)

if __name__ == "__main__":
    main()