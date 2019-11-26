import cv2 
import pygame
import numpy as np
from pygame import display



# pygame parametres 

display_width = 640
display_height = 360

gameDisplay = display.set_mode((display_width,display_height))


def frameProcessing(image):

    #print(image.shape)
    heigh, wight, depth = image.shape
    image =cv2.resize(image,(wight//3,heigh//3))
    #cv2.imshow('Frame',image)
    #cv2.imshow('Frame',image)   



    # if you want to use pygame for showing the vide!
    surf = pygame.surfarray.make_surface(np.swapaxes(image,0,1))
    gameDisplay.blit(surf,(0,0))
    display.update()
    
    print(image.shape)






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


