"""
K-pop Fancam Face Extractor
Extracts, aligns, and normalizes faces from YouTube fancam videos.

Input: Give description of Kpop-Idol name
"""
#Thanks to Andrew for writing the majority of this script


TF_ENABLE_ONEDNN_OPTS=0

import os
import sys
from datetime import datetime
import cv2
import numpy as np
import yt_dlp
import mediapipe as mp
from pathlib import Path


sys.stdout.reconfigure(encoding="utf-8")

#Defaults to Wonhee, Manually Change for Now.
def find_m2_fancams(query="원희", aliases=("원희", "Wonhee"), max_videos=50):
        url = f"https://www.youtube.com/@MnetM2/search?query={query}"

        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
            "ignoreerrors": True,
            "playlistend": max_videos,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        urls = []
        for entry in info.get("entries", []):
            if not entry:
                continue
            title = entry.get("title") or ""
            if any(a.lower() in title.lower() for a in aliases):
                urls.append((title, f"https://www.youtube.com/watch?v={entry['id']}"))
        return urls
class FaceExtractor:
    def __init__(
        self,
        output_size=512,
        frame_skip=5,
        jpg_quality=95,
        min_face_size=100,
        blur_threshold=100,
        min_detection_confidence=0.7,
        face_margin=0.6
    ):
        """
        Initialize the face extractor.
       
        Args:
            output_size: Final image dimensions (square)
            frame_skip: Process every Nth frame
            jpg_quality: JPEG quality (1-100)
            min_face_size: Minimum face height in pixels to accept
            blur_threshold: Laplacian variance below this = blurry
            min_detection_confidence: MediaPipe detection threshold (0-1)
            face_margin: Extra margin around face (0.5 = 50% padding)
        """
        self.output_size = output_size
        self.frame_skip = frame_skip
        self.jpg_quality = jpg_quality
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.min_detection_confidence = min_detection_confidence
        self.face_margin = face_margin
       
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
    


    



    def download_video(self, url: str, output_path: str) -> bool:
        """Download YouTube video in best quality."""
        print(f"Downloading video from: {url}")
       
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print("Download complete!")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
   
    def is_blurry(self, image: np.ndarray) -> bool:
        """Check if image is blurry using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold
   
    def get_eye_positions(self, face_landmarks, image_width, image_height):
        """
        Extract eye center positions from MediaPipe face landmarks.
       
        MediaPipe landmark indices:
        - Left eye: 33, 133, 159, 145 (corners and top/bottom)
        - Right eye: 362, 263, 386, 374
        - Left eye center approximation: average of 33 and 133
        - Right eye center approximation: average of 362 and 263
        """
        # Left eye landmarks
        left_eye_inner = face_landmarks.landmark[133]
        left_eye_outer = face_landmarks.landmark[33]
       
        # Right eye landmarks  
        right_eye_inner = face_landmarks.landmark[362]
        right_eye_outer = face_landmarks.landmark[263]
       
        # Calculate eye centers
        left_eye_x = (left_eye_inner.x + left_eye_outer.x) / 2 * image_width
        left_eye_y = (left_eye_inner.y + left_eye_outer.y) / 2 * image_height
       
        right_eye_x = (right_eye_inner.x + right_eye_outer.x) / 2 * image_width
        right_eye_y = (right_eye_inner.y + right_eye_outer.y) / 2 * image_height
       
        return (left_eye_x, left_eye_y), (right_eye_x, right_eye_y)
   
    def is_frontal_face(self, face_landmarks, image_width, image_height) -> bool:
        """
        Check if face is roughly frontal using landmark positions.
        Compares nose position relative to eye centers.
        """
        # Get key landmarks
        nose_tip = face_landmarks.landmark[4]
        left_eye, right_eye = self.get_eye_positions(face_landmarks, image_width, image_height)
       
        # Eye center
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_width = abs(right_eye[0] - left_eye[0])
       
        if eye_width < 10:  # Eyes too close, likely profile
            return False
       
        # Nose tip position
        nose_x = nose_tip.x * image_width
       
        # Check if nose is roughly centered between eyes
        nose_offset = abs(nose_x - eye_center_x) / eye_width
       
        # Allow up to 30% offset (roughly 30 degree turn)
        return nose_offset < 0.3
   
    def align_face(self, image: np.ndarray, left_eye: tuple, right_eye: tuple) -> np.ndarray:
        """Rotate image so eyes are horizontal."""
        # Calculate angle between eyes
        delta_y = right_eye[1] - left_eye[1]
        delta_x = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
       
        # Get image center and rotation matrix
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
       
        # Rotate image
        aligned = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC)
       
        return aligned, matrix
   
    def transform_point(self, point: tuple, matrix: np.ndarray) -> tuple:
        """Transform a point using rotation matrix."""
        px, py = point
        new_x = matrix[0, 0] * px + matrix[0, 1] * py + matrix[0, 2]
        new_y = matrix[1, 0] * px + matrix[1, 1] * py + matrix[1, 2]
        return (new_x, new_y)
   
    def crop_and_normalize(
        self,
        image: np.ndarray,
        left_eye: tuple,
        right_eye: tuple
    ) -> np.ndarray:
        """Crop face with margin and resize to target size."""
       
        # Calculate face region based on eye positions
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        eye_width = abs(right_eye[0] - left_eye[0])
       
        # Estimate face size from eye width (eyes are roughly 1/3 of face width)
        face_width = eye_width * 3
        face_height = face_width * 1.3  # Faces are slightly taller than wide
       
        # Face center is slightly below eye center
        face_center_y = eye_center_y + face_height * 0.1
       
        # Add margin
        total_size = max(face_width, face_height) * (1 + self.face_margin)
        half_size = int(total_size / 2)
       
        # Calculate crop boundaries
        center_x = int(eye_center_x)
        center_y = int(face_center_y)
       
        crop_top = center_y - half_size
        crop_bottom = center_y + half_size
        crop_left = center_x - half_size
        crop_right = center_x + half_size
       
        # Handle edge cases with padding
        pad_top = max(0, -crop_top)
        pad_bottom = max(0, crop_bottom - image.shape[0])
        pad_left = max(0, -crop_left)
        pad_right = max(0, crop_right - image.shape[1])
       
        # Adjust crop boundaries
        crop_top = max(0, crop_top)
        crop_bottom = min(image.shape[0], crop_bottom)
        crop_left = max(0, crop_left)
        crop_right = min(image.shape[1], crop_right)
       
        # Crop
        face = image[crop_top:crop_bottom, crop_left:crop_right]
       
        # Check if crop is valid
        if face.size == 0:
            return None
       
        # Pad if necessary
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            face = cv2.copyMakeBorder(
                face,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT
            )
       
        # Resize to target size
        face = cv2.resize(
            face,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_LANCZOS4
        )
       
        return face
   
    def process_frame(self, frame: np.ndarray, face_detection, face_mesh) -> np.ndarray | None:
        """
        Process a single frame and extract face if valid.
        Returns cropped face image or None if no valid face found.
        """
        h, w = frame.shape[:2]
       
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        # Detect faces
        detection_results = face_detection.process(rgb_frame)
       
        if not detection_results.detections:
            return None
       
        # Must have exactly one face
        if len(detection_results.detections) != 1:
            return None
       
        detection = detection_results.detections[0]
       
        # Check detection confidence
        if detection.score[0] < self.min_detection_confidence:
            return None
       
        # Get bounding box
        bbox = detection.location_data.relative_bounding_box
        face_height = int(bbox.height * h)
       
        # Check minimum face size
        if face_height < self.min_face_size:
            return None
       
        # Get detailed face landmarks using Face Mesh
        mesh_results = face_mesh.process(rgb_frame)
       
        if not mesh_results.multi_face_landmarks:
            return None
       
        if len(mesh_results.multi_face_landmarks) != 1:
            return None
       
        face_landmarks = mesh_results.multi_face_landmarks[0]
       
        # Check if face is frontal
        if not self.is_frontal_face(face_landmarks, w, h):
            return None
       
        # Get eye positions
        left_eye, right_eye = self.get_eye_positions(face_landmarks, w, h)
       
        # Align face (rotate so eyes are horizontal)
        aligned, rotation_matrix = self.align_face(frame, left_eye, right_eye)
       
        # Transform eye positions to aligned image
        aligned_left_eye = self.transform_point(left_eye, rotation_matrix)
        aligned_right_eye = self.transform_point(right_eye, rotation_matrix)
       
        # Crop and normalize
        face_crop = self.crop_and_normalize(aligned, aligned_left_eye, aligned_right_eye)
       
        if face_crop is None:
            return None
       
        # Check for blur
        if self.is_blurry(face_crop):
            return None
       
        return face_crop
   
    def extract_from_video(self, video_path: str, output_dir: str) -> int:
        """
        Extract faces from video file.
       
        Args:
            video_path: Path to video file
            output_dir: Directory to save face images
           
        Returns:
            Number of faces saved
        """
      

        Path(output_dir).mkdir(parents=True, exist_ok=True)
       
        cap = cv2.VideoCapture(video_path)
       
        if not cap.isOpened():
            print("Could not open video")
            return -1
       
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
       
        frame_count = 0
        saved_count = 0
       
        # Initialize MediaPipe models
        with self.mp_face_detection.FaceDetection(
            model_selection=1,  
            min_detection_confidence=self.min_detection_confidence
        ) as face_detection, self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5
        ) as face_mesh:
           
            while True:
                ret, frame = cap.read()
               
                if not ret:
                    break
               
                # Progress update
                if frame_count % 500 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames}) - Saved: {saved_count}")
               
                # Skip frames according to frame_skip setting
                if frame_count % self.frame_skip == 0:
                    face_image = self.process_frame(frame, face_detection, face_mesh)
                   
                    if face_image is not None:
                        # Save face image
                        filename = f"face_{saved_count:06d}.jpg"
                        filepath = os.path.join(output_dir, filename)
                       
                        cv2.imwrite(
                            filepath,
                            face_image,
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
                        )
                        saved_count += 1
               
                frame_count += 1
       
        cap.release()
       
        print()
        print(f"Extraction complete!")
        print(f"   Frames processed: {frame_count}")
        print(f"   Faces saved: {saved_count}")
        print(f"   Output directory: {output_dir}")
       
        return saved_count
   
    def run(self, youtube_url: str, output_base_dir: str = "output") -> str:
        """
        Main entry point - download video and extract faces.
       
        Args:
            youtube_url: YouTube video URL
            output_base_dir: Base directory for outputs
           
        Returns:
            Path to output directory
        """
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       
        # Setup paths
        temp_video = f"temp_video_{timestamp}.mp4"
        output_dir = os.path.join(output_base_dir, f"faces_{timestamp}")
       
       
        if not self.download_video(youtube_url, temp_video):
            return ""
       
        # Extract faces
        saved_count = self.extract_from_video(temp_video, output_dir)
        '''Init here
        '''
       
        # Cleanup temp video
        if os.path.exists(temp_video):
            os.remove(temp_video)
            print(f"Cleaned up temporary video file")
        
        # Calculate storage used
        if saved_count > 0:
            total_size = sum(
                os.path.getsize(os.path.join(output_dir, f))
                for f in os.listdir(output_dir)
                if f.endswith('.jpg')
            )
            size_mb = total_size / (1024 * 1024)
        else:
            size_mb = 0
       
        print()
        print("=" * 50)
        print("Summary")
        print("=" * 50)
        print(f"   Faces extracted: {saved_count}")
        print(f"   Storage used: {size_mb:.2f} MB")
        print(f"   Output: {output_dir}")
        print()
       
        return output_dir


def main():
   
    # Configuration
    extractor = FaceExtractor(
        output_size=512,                
        frame_skip=5,                  
        jpg_quality=95,               
        min_face_size=100,             
        blur_threshold=100,            
        min_detection_confidence=0.7,  
        face_margin=0.6                
    )
    urls = []
    for title, u in find_m2_fancams("원희 Fancam"):
        urls.append(u)
 
    for url in urls:
        output_dir = extractor.run(url)
   
        if output_dir:
            print(f" Done! Check your faces in: {output_dir}")


if __name__ == "__main__":
    main()