import numpy as np
import cv2


class FeatureExtractor():
    # init block defind the split parameters
    def __init__(self):
        self.indexParams = dict(algorithm=0, trees=5)
        self.searchParms = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.indexParams, self.searchParms)
        self.last = None
        self.GX = 2
        self.GY = 2
        super().__init__()

    # extract keypoints by dividing the frame to get a distributed keyPoints
    def extract(self, img, descriptor):
        totalKps = []
        totalDes = None
        count = 0
        goodMatches = []
       

        # print(totalDes.shape)

        dy = img.shape[0]//self.GY
        dx = img.shape[1]//self.GX
        for i in range(0, img.shape[0], 3*dy//4):
            for j in range(0, img.shape[1], 3*dx//4):
                # cv2.imshow('image',img[i:i+dy,j:j+dx])

                # time.sleep(0.1)
                kp, des = descriptor.detectAndCompute(
                    img[i:i+dy, j:j+dx], None)
                
                for p in kp:
                    # print(p.pt)
                    p.pt = (p.pt[0]+j, p.pt[1]+i)
                    # print(p.pt)
                totalKps.append(kp)
                if (des is not None):
                    count += des.shape[0]
                    if (totalDes is None):
                        totalDes = des
                    else:
                        totalDes = np.append(totalDes, des, axis=0)
        totalKps = sum(totalKps,[])
        if (self.last is not None):

            totalDes = np.float32(totalDes)
            self.last['des'] = np.float32(self.last['des'])

            knn_matches = self.flann.knnMatch(totalDes, self.last['des'], k=2)

            ratio_theashold = 0.72
            #print(knn_matches)
            # keeping only good matches
            for m, n in knn_matches:
                if m.distance < ratio_theashold * n.distance:
                    goodMatches.append((totalKps[m.queryIdx],self.last['kps'][m.trainIdx]))
                 

        # print(goodMatches)
        """
        print(count)
        print(totalDes.shape)
        print(len(sum(totalKps,[])))
        """
        self.last = {'kps': totalKps, 'des': totalDes}
        print(len(goodMatches), 'matches')
        return totalKps, totalDes, goodMatches
