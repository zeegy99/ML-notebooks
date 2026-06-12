import os
import csv
import PIL 
import cv2 
import sys 

sys.stdout.reconfigure(encoding="utf-8")

def photos_from_video(vid_path, output_dir = 'DDPM/photos'):
    video_path = vid_path 
    vidcap = cv2.VideoCapture(video_path)

    video_name = vid_path[vid_path.find('/videos/') + len('/videos/'):]
    
    output_dir = output_dir + f'_{video_name}'

    os.makedirs(output_dir, exist_ok=True)


    success, frame = vidcap.read()
    count = 0


    while success:
        # Save the frame as a JPG image
        photo_path = os.path.join(output_dir, f'frame_{count:04d}.jpg')
        cv2.imwrite(photo_path, frame)
        
        # Read next frame
        success, frame = vidcap.read()
        count += 1

    print(f"Successfully extracted {count} photos to '{output_dir}/'")
    vidcap.release()

if __name__ == '__main__':
    video_path = 'DDPM/videos/R U Next？ 원희 WONHEE l 3R rehearsal FanCam.webm'

    photos_from_video(video_path)