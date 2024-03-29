import cv2
import time
import pygame
import numpy as np
from extractors import FeatureExtractor as fExt
from pygame import display


"""
# pygame parametres 

display_width = 640
display_height = 360

gameDisplay = display.set_mode((display_width,display_height))

"""
fe = fExt.FeatureExtractor()


F = 50


def frameProcessing(image, fe):

    # print(image.shape)
    heigh, wight, depth = image.shape
    image = cv2.resize(image, (wight//3, heigh//3))

    K = np.array([[F,0,image.shape[0]//2],[0,F,image.shape[1]//2],[0,0,1]])
    # cv2.imshow('Frame',image)
    # cv2.imshow('Frame',image)

    # if you want to use sift or surf decomment these lines
    #sift = cv2.xfeatures2d.SIFT_create()
    #surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=1500)
    kp, des, matches = fe.extract(image, orb,K)

    # print(kp)
    #keypoints_sift, _ = orb.detectAndCompute(image,None)
    # drawing keypoints
    img = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))
    if (len(matches) != 0):
        # print(matches)

        for i, j in matches:
            x1, y1 = fe.deNormalize(i)
            x2, y2 = fe.deNormalize(j)
            img = cv2.line(img, (x1, y1), (x2, y2), color=(0, 0, 255))

    cv2.imshow('Frame', img)


"""
    # if you want to use pygame for showing the video!
    surf = pygame.surfarray.make_surface(np.swapaxes(image,0,1))
    gameDisplay.blit(surf,(0,0))
    display.update()
    
    print(image.shape)
"""

if __name__ == '__main__':
   cap = cv2.VideoCapture("./videos/test_countryroad.mp4")


   if (cap.isOpened() == False):
       print("Error opening")

   while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        frameProcessing(frame, fe)

        if (cv2.waitKey(25) & 0xFF == ord("q")):
            break
    else:
        break

   cap.release()

   cv2.destroyAllWindows()
