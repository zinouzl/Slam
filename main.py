import cv2 



cap = cv2.VideoCapture("./videos/test_countryroad.mp4")


if (cap.isOpened()==False):
    print("Error opening")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        #cv2.imshow('Frame',frame)
        print('h')

        if (cv2.waitKey(25) & 0xFF == ord("q")):
            break
    else:
        break

cap.release()

cv2.destroyAllWindows()


print("hello word")