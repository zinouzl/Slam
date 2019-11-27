import cv2 
import time
import pygame
import numpy as np
from pygame import display


"""
# pygame parametres 

display_width = 640
display_height = 360

gameDisplay = display.set_mode((display_width,display_height))

"""

class FeatureExtraxture():
    def __init__(self):
        
        self.GX = 2
        self.GY = 2
        super().__init__()

    def extract(self,img,descriptor):
        totalKps =[]
        dy  = img.shape[0]//self.GY
        dx  = img.shape[1]//self.GX
        for i in range(0,img.shape[0],dy):
            for j in range(0,img.shape[1],dx):
                #cv2.imshow('image',img[i:i+dy,j:j+dx])
                #qqqqqqqqqqqqqqqqq
                #time.sleep(0.1)
                kp,_ = descriptor.detectAndCompute(img[i:i+dy,j:j+dx],None)
                for p in kp:
                    #print(p.pt)
                    p.pt = (p.pt[0]+j,p.pt[1]+i)
                    #print(p.pt)
                totalKps.append(kp)
                

        
        #print(kp)
        return totalKps





def frameProcessing(image):
    fe = FeatureExtraxture()
    #print(image.shape)
    heigh, wight, depth = image.shape
    image =cv2.resize(image,(wight//3,heigh//3))
    #cv2.imshow('Frame',image)
    #cv2.imshow('Frame',image)   
    sift = cv2.xfeatures2d.SIFT_create()
    #surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=1500)
    kp = fe.extract(image,orb)
    kp = sum(kp,[])
    print(kp)
    #keypoints_sift, _ = orb.detectAndCompute(image,None)
    img = cv2.drawKeypoints(image,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

     
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
        
        frameProcessing(frame)

        
        

        if (cv2.waitKey(25) & 0xFF == ord("q")):
            break
    else:
        break

cap.release()

cv2.destroyAllWindows()


