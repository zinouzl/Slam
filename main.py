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

def frameProcessing(image,fe):
    
    #print(image.shape)
    heigh, wight, depth = image.shape
    image =cv2.resize(image,(wight//3,heigh//3))
    #cv2.imshow('Frame',image)
    #cv2.imshow('Frame',image)   
    sift = cv2.xfeatures2d.SIFT_create()
    #surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=1500)
    kp,des = fe.extract(image,orb)
    kp = sum(kp,[])
    #print(kp)
    #keypoints_sift, _ = orb.detectAndCompute(image,None)
    img = cv2.drawKeypoints(image,kp,None,color=(0,255,0))

     
    cv2.imshow('Frame',img)


"""
    # if you want to use pygame for showing the vide!
    surf = pygame.surfarray.make_surface(np.swapaxes(image,0,1))
    gameDisplay.blit(surf,(0,0))
    display.update()
    
    print(image.shape)
"""





cap = cv2.VideoCapture("./videos/test_countryroad.mp4")


if (cap.isOpened()==False):
    print("Error opening")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        
        frameProcessing(frame,fe)

        
        

        if (cv2.waitKey(25) & 0xFF == ord("q")):
            break
    else:
        break

cap.release()

cv2.destroyAllWindows()


