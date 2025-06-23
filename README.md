# face-detection_img-video.ipynb
✅ STEP 1: Install OpenCV
python
Copy
Edit
!pip install opencv-python
!pip install opencv-python-headless
opencv-python: Main OpenCV package (computer vision functions).

opencv-python-headless: No GUI; required in Google Colab (where imshow() is blocked).

✅ STEP 2: Import Libraries
python
Copy
Edit
import cv2
import matplotlib.pyplot as plt
cv2: Main OpenCV library for image and video operations.

matplotlib.pyplot: Optional – for plotting if needed.

✅ STEP 3: Download Haar Cascade
python
Copy
Edit
!wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
Downloads pre-trained Haar cascade XML file used to detect faces (rule-based model from OpenCV).

✅ STEP 4: Upload Image to Colab
python
Copy
Edit
from google.colab import files
uploaded = files.upload()
Opens a file upload box in Colab.

After upload, the file is saved to /content/filename.jpg.

✅ STEP 5: Load Image and Detect Face
python
Copy
Edit
from google.colab.patches import cv2_imshow  # Colab-safe imshow

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cv2_imshow: Used instead of cv2.imshow() in Colab.

CascadeClassifier: Loads the face detection XML model.

python
Copy
Edit
img = cv2.imread('/content/WhatsApp Image 2025-06-15 at 8.16.46 PM.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Reads the uploaded image.

Converts it to grayscale (required for Haar cascade detection).

✅ STEP 6: Detect Faces
python
Copy
Edit
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100))
Detects faces in the grayscale image.

scaleFactor=1.1: Shrinks image for pyramid detection.

minNeighbors=3: Higher = stricter face detection.

minSize=(100, 100): Detect only faces at least 100x100 pixels in size.

✅ STEP 7: Draw Face Box
python
Copy
Edit
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
Loops through all detected faces.

Draws a green rectangle with thickness 2.

python
Copy
Edit
cv2_imshow(img)
Displays the image with rectangle in Colab.

✅ STEP 8: Upload a Video
python
Copy
Edit
uploaded = files.upload()
video_path = list(uploaded.keys())[0]
Uploads video file.

Gets filename into video_path.

✅ STEP 9: Face Detection on Video
python
Copy
Edit
cap = cv2.VideoCapture('/content/WhatsApp Video 2025-06-15 at 8.16.46 PM.mp4')
Opens the uploaded video for reading.

python
Copy
Edit
frame_count = 0
max_frames = 30
Only shows 30 frames (prevents too many outputs in Colab).

✅ STEP 10: Read and Process Video Frame-by-Frame
python
Copy
Edit
while True:
    ret, frame = cap.read()
    if not ret or frame_count >= max_frames:
        break
Reads video frames.

Stops when video ends or 30 frames are processed.

✅ STEP 11: Convert, Detect, and Draw Boxes
python
Copy
Edit
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
Converts each frame to grayscale.

Detects faces.

python
Copy
Edit
for (x, y, w, h) in faces:
    padding = 20
    x_pad = max(x - padding, 0)
    y_pad = max(y - padding, 0)
    w_pad = w + 2 * padding
    h_pad = h + 2 * padding
    cv2.rectangle(frame, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (0, 255, 0), 2)
Draws a larger green rectangle around each face by adding padding.

✅ STEP 12: Show Each Frame (Colab-safe)
python
Copy
Edit
cv2_imshow(frame)
frame_count += 1
Shows the current processed frame with rectangles.

Increases frame_count until 30.

python
Copy
Edit
cap.release()
Releases the video resource when done.

