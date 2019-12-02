import numpy as np
import cv2
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform
from skimage.measure import ransac

def add_ones(x):
    x = np.concatenate((x,np.ones((x.shape[0],1))),axis=1)
    return x

class FeatureExtractor():
    # init block defind the split parameters
    def __init__(self):
        self.indexParams = dict(algorithm=0, trees=5)
        self.searchParms = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.indexParams, self.searchParms)
        self.last = None
        self.GX = 2
        self.GY = 2
        self.K = None
        self.fEstm = []
        super().__init__()


    # denormalize point:
    def deNormalize(self,p):
        return (round(p.pt[0] ),round(p.pt[1]))



    # extract keypoints by dividing the frame to get a distributed keyPoints
    def extract(self, img, descriptor,K):
        self.K = K
        self.Kinv = np.linalg.inv(K)
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
            # F = 1500
            # tKeypoints[:,0] -= img.shape[0]//2
            # tKeypoints[:,1] -= img.shape[1]//2


            print(lastKeypoints.shape)

            tKeypoints  = np.dot(add_ones(tKeypoints),self.Kinv)


            # tKeypoints += F
            # lastKeypoints[:,0] -= img.shape[0]//2
            # lastKeypoints[:,1] -= img.shape[1]//2
            lastKeypoints = np.dot(add_ones(lastKeypoints),self.Kinv)
            # lastKeypoints += F
            #tKeypoints = np.dot(tKeypoints,K)
            #lastKeypoints = np.dot(lastKeypoints,K)
            #  print(lastKeypoints.shape)
            

            model, inliers = ransac((tKeypoints[:,0:2],
                                     lastKeypoints[:,0:2]),
                                    EssentialMatrixTransform,
                                    min_samples=8,
                                    
                                    residual_threshold=0.01,
                                    max_trials=200)
            # print(inliers)
            # print(inliers.shape)
            # print(lastKeypoints.shape)
            goodMatchesLast = goodMatches[inliers, 1]
            goodMatchesT = goodMatches[inliers, 0]

            
            

            filtredGoodMatches = [(i, j)
                                  for i, j in zip(goodMatchesT, goodMatchesLast)]
            # print(filtredGoodMatches)
            R,t = extractRt(model.params)
        # print(goodMatches)
        """
        print(count)
        print(totalDes.shape)
        print(len(sum(totalKps,[])))
        """
        self.last = {'kps': totalKps, 'des': totalDes}
        print(len(filtredGoodMatches), 'matches')
        return totalKps, totalDes, filtredGoodMatches


def extractRt(E):
    S,v,D = np.linalg.svd(E)

    print(S.shape)
    print(v.shape)
    print(D.shape)
            #print(v)
    # self.fEstm.append(np.sqrt(2)/((v[0]+v[1])/2))
    diag  = np.array(([1,0,0],[0,1,0],[0,0,0]))

    newE = np.dot(np.dot(S,diag),D.T)
    S,v,D = np.linalg.svd(newE)

    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    if (np.linalg.det(D)<0):
        D *= -1.0

    #print(np.mean(self.fEstm))
    R = np.dot(np.dot(S,W),D)
    if np.sum(R.diagonal())<0:
        R = np.dot(np.dot(S,W.T),D)
    #R = s * W * d

    t = S[:,2]
    print(R)

    return R,t