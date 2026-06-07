    #Measuring image similarity in Python using ORB 

#Chose ORB because this accurately detects rotations as the same image, which happens very often


import cv2 

# print('gurt')
#Works well with images of different dimensions
def orb_sim(img1, img2):
    # SIFT is no longer available in cv2 so using ORB
    orb = cv2.ORB_create()

    # detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # define the bruteforce matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    #perform matches. 
    matches = bf.match(desc_a, desc_b)
    #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
    similar_regions = [i for i in matches if i.distance < 50]  
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

for i in range(0, 10): 
    for j in range(i, 10):

        img01 = cv2.imread(f'../../output/faces_20260606_193033/face_00000{i}.jpg', 0)
        img02 = cv2.imread(f'../../output/faces_20260606_193033/face_00000{j}.jpg', 0)
        print(f"This is the orb sim between {i} and {j}",  orb_sim(img01, img02))