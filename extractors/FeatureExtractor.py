import numpy as np
import cv2
from skimage.transform import FundamentalMatrixTransform
from skimage.measure import ransac


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
        filtredGoodMatches = []

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
        totalKps = sum(totalKps, [])
        if (self.last is not None):

            totalDes = np.float32(totalDes)
            self.last['des'] = np.float32(self.last['des'])

            knn_matches = self.flann.knnMatch(totalDes, self.last['des'], k=2)

            ratio_theashold = 0.78
            # print(knn_matches)
            # keeping only good matches
            for m, n in knn_matches:
                if m.distance < ratio_theashold * n.distance:
                    goodMatches.append(
                        (totalKps[m.queryIdx], self.last['kps'][m.trainIdx]))

            # filtring data using ransac over fundamental matrix transform
            goodMatch = np.array([[i.pt, j.pt] for i, j in goodMatches])
            # print(goodMatch.shape)
            goodMatches = np.array(goodMatches)
            #tKeypoint = [x[0] for x in goodMatches]
            #lastKeypoints = [x[1] for x in goodMatches]
            
            tKeypoints = goodMatch[:, 0]
            lastKeypoints = goodMatch[:, 1]

            # Normalize : moving to the center by dividing over 2
            
            # print(lastKeypoints.shape)

            model, inliers = ransac((tKeypoints,
                                     lastKeypoints),
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=0.9,
                                    max_trials=100)
            # print(inliers)
            # print(inliers.shape)
            # print(lastKeypoints.shape)
            goodMatchesLast = goodMatches[inliers, 1]
            goodMatchesT = goodMatches[inliers, 0]

            filtredGoodMatches = [(i, j)
                                  for i, j in zip(goodMatchesT, goodMatchesLast)]
            # print(filtredGoodMatches)

        # print(goodMatches)
        """
        print(count)
        print(totalDes.shape)
        print(len(sum(totalKps,[])))
        """
        self.last = {'kps': totalKps, 'des': totalDes}
        print(len(filtredGoodMatches), 'matches')
        return totalKps, totalDes, filtredGoodMatches
