import cv2
import numpy as np

# Load images for helmet and mustache
helmet = cv2.imread('mascot.png', cv2.IMREAD_UNCHANGED)
mustache = cv2.imread('stache.png', cv2.IMREAD_UNCHANGED)

if helmet is None or mustache is None:
    print("Unable to load images... please check images are in the same director as the program.")
    exit()

# Default to showing helmet
current_image = helmet
use_helmet = True

# Load face detector using opencv's "Haar Cascade" --> Fast and works in real time
#It is a pretained model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam capture
cap = cv2.VideoCapture(0)

def overlay(img, overlay, x, y):
    
    h, w = overlay.shape[0], overlay.shape[1]

    # Crop overlay and background if overlay goes out of bounds
    if y + h > img.shape[0] or x + w > img.shape[1] or x < 0 or y < 0:
        return
    #Split into RGB
    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background = img[y:y+h, x:x+w]

    img[y:y+h, x:x+w] = (1 - mask) * background + mask * overlay_img

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #Converts frame into grayscale to improve face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
 
    for (x, y, w, h) in faces:
        if use_helmet:
            # Resize and position helmet slightly above the head
            helmet_resized = cv2.resize(helmet, (int(w*1.3), int(h*1.3)))
            hx = x - int(w * 0.2)  # adjust to center helmet
            hy = y - int(h * 0.2)  # raise above face, can be adjusted based of user
            overlay(frame, helmet_resized, hx, hy)
        else:
            # Resize and position mustache below nose 
            mustache_width = int(w * 0.6) # can be adjusted based of user
            mustache_height = int(h * 0.2)  # can be adjusted based of user
            mustache_resized = cv2.resize(mustache, (mustache_width, mustache_height))
            mx = x + int((w - mustache_width) / 2)
            my = y + int(h * 0.65)  # lower part of face, can be tuned based off the user
            overlay(frame, mustache_resized, mx, my)

    cv2.imshow("Face", frame)
    #Check for key presses
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        use_helmet = not use_helmet
        current_image = helmet if use_helmet else mustache

cap.release()
cv2.destroyAllWindows()