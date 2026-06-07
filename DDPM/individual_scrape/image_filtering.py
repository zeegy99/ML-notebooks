import cv2
import os 

def get_blur_score(image_path):
    """
    Calculates the focus measure using the variance of the Laplacian.
    Higher values mean sharper/less blurred images.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
        
    # Apply the Laplacian operator and return its variance
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


threshold = 20
for i in range((5235 + 1)):
    frame_number = str(i).zfill(4)
    # print(frame_number)
    image_path = f"DDPM/photos/frame_{frame_number}.jpg"
    blur_score = get_blur_score(image_path)
    if blur_score < threshold:
        os.remove(image_path)


