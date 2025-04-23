# Virtual_Try_On_Glasses
Summary of the Code:
This code is a Streamlit-based virtual try-on application that allows users to upload an image and overlay virtual glasses (or any other accessory). The glasses are placed on the user's face using MediaPipe's Face Mesh model, which detects facial landmarks. The glasses image is then positioned and resized based on the eye positions of the detected face.
Here’s how the app works:
1.	User Uploads Image: The user uploads an image through a file uploader.
2.	Face Detection: The MediaPipe Face Mesh model detects the facial landmarks, particularly the eyes, in the uploaded image.
3.	Glasses Overlay: The code then overlays a glasses image on top of the detected eye region using the landmarks.
4.	Output: The processed image (with the glasses overlaid) is displayed back to the user.
Explanation of the Libraries Used in the Code:
1. Streamlit (import streamlit as st)
•	Purpose: Streamlit is a Python library used to quickly build interactive web apps with minimal code.
•	Functions used:
o	st.title(): Sets the title of the web page.
o	st.file_uploader(): Allows the user to upload an image file.
o	st.image(): Displays an image on the web page.
•	Streamlit is the framework that runs the web interface for this application.
2. OpenCV (import cv2)
•	Purpose: OpenCV (Open Source Computer Vision Library) is a popular library used for computer vision tasks, such as image processing, face detection, object detection, etc.
•	Functions used:
o	cv2.imread(): Loads the image (in this case, the glasses image) from a file.
o	cv2.cvtColor(): Converts the image from one color space to another (e.g., RGB to BGR for OpenCV compatibility).
o	cv2.resize(): Resizes the glasses image to fit the face's eye region.
o	cv2.copy(): Creates a copy of the image to avoid modifying the original one.
•	OpenCV is primarily used for reading, manipulating, and displaying images in this code.
3. NumPy (import numpy as np)
•	Purpose: NumPy is a library for numerical operations in Python, often used to handle arrays, matrices, and mathematical functions.
•	Functions used:
o	np.linalg.norm(): Calculates the Euclidean distance between two points (used to determine the distance between the eyes to estimate glasses size).
o	np.array(): Converts data into an array (used to handle image data and landmarks).
•	NumPy is used to perform mathematical operations, like computing distances for resizing the glasses, and to work with image data as arrays.
4. MediaPipe (import mediapipe as mp)
•	Purpose: MediaPipe is a cross-platform framework by Google designed for building multimodal applied machine learning pipelines. It provides models for various tasks, including face detection, hand tracking, pose estimation, etc.
•	Functions used:
o	mp.solutions.face_mesh.FaceMesh(): Loads the Face Mesh model, which detects facial landmarks (such as the eyes, nose, mouth, etc.) in an image.
o	mp.solutions.drawing_utils.draw_landmarks(): Draws landmarks on the detected face in the image for visualization.
o	face_landmarks.landmark: Provides access to the detected facial landmarks (x, y coordinates).

•	MediaPipe is crucial for detecting facial landmarks and processing the face mesh in the image. This allows the glasses to be accurately placed on the face based on the detected eye positions.
6. PIL (Python Imaging Library) (from PIL import Image)
•	Purpose: PIL (now known as Pillow) is a library used for opening, manipulating, and saving images in various formats.
•	Functions used:
o	Image.open(): Opens an image file for manipulation.
•	PIL is used to load the uploaded image and process it before applying the overlay.
Detailed Explanation of the Code's Functionality:
1. Upload Image:
•	The file_uploader widget in Streamlit allows the user to upload an image.
•	The uploaded image is loaded using Pillow (Image.open()), and then converted into a format suitable for OpenCV using cv2.cvtColor().
2. Face Mesh Detection with MediaPipe:
•	The FaceMesh model from MediaPipe is used to detect landmarks on the face. These landmarks are precise points on the face, such as the eyes, eyebrows, nose, etc.
•	The face mesh model processes the image to identify the positions of the eyes (landmark points 33 and 133, representing the left and right eye centers).
3. Overlay Glasses:
•	The glasses image is loaded using OpenCV (cv2.imread()), and its dimensions are adjusted based on the distance between the eyes.
•	The glasses image is resized to fit the face using the calculated width and height.
•	The overlay_glasses() function places the glasses image onto the detected region of the face using the calculated position and size.
o	The function checks if the glasses image has an alpha channel (transparency). If so, it uses alpha blending to overlay the glasses on the face.
4. Displaying the Image:
•	After the glasses are successfully overlaid on the face, the modified image is displayed back to the user using st.image().
Possible Issues/Improvements:
•	Glasses Alignment: The current implementation uses fixed landmarks for the eye positions. Depending on the face orientation, more complex logic may be needed to adjust for different angles or facial orientations.
•	Glasses Image Transparency: The overlaying technique checks for transparency in the glasses image, which is crucial for smooth integration with the face.
This application can be enhanced by adding more customization, such as allowing the user to upload different glasses images or detect multiple faces in a single image.

# Team Members
1. Rudra Dholariya
2. Munjal Vyas
3. Dhavalsinh Vaghela
