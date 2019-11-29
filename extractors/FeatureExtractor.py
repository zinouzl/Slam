class FeatureExtractor():
    def __init__(self):
        
        self.GX = 2
        self.GY = 2
        super().__init__()

    def extract(self,img,descriptor):
        totalKps =[]
        dy  = img.shape[0]//self.GY
        dx  = img.shape[1]//self.GX
        for i in range(0,img.shape[0],3*dy//4):
            for j in range(0,img.shape[1],3*dx//4):
                #cv2.imshow('image',img[i:i+dy,j:j+dx])
            
                #time.sleep(0.1)
                kp= descriptor.detect(img[i:i+dy,j:j+dx],None)
                for p in kp:
                    #print(p.pt)
                    p.pt = (p.pt[0]+j,p.pt[1]+i)
                    #print(p.pt)
                totalKps.append(kp)
                

        
        #print(kp)
        return totalKps