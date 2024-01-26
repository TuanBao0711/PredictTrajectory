from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d

from boxmot import DeepOCSORT
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
from boxmot import StrongSORT

import time
import numpy as np
from scipy.stats import zscore



import cv2

def distance(x1, y1, x2, y2):

  distance = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

  return distance


tracker = BYTETracker()

# SuperPoint+LightGlue
# extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
# matcher = LightGlue(features='superpoint').eval().cuda()

extractor = SIFT(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='sift').eval().cuda()

# extractor = SIFT(max_num_keypoints=None).eval().cuda()
# matcher = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).eval().cuda()


# imgObject = cv2.imread("C:/Users/TuanBao/Desktop/My_Docs/CNTT/imgMatching/lightGlue/mig/plane1.png")
imgObject = cv2.imread("C:/Users/TuanBao/Desktop/My_Docs/CNTT/imgMatching/lightGlue/mig/plane1.png")
# imgObject = cv2.fastNlMeansDenoisingColored(imgObject,None,10,10,7,21)
imgObject = cv2.cvtColor(imgObject, cv2.COLOR_BGR2GRAY)
# cv2.imshow('object',imgObject)
image0 = numpy_image_to_torch(imgObject).cuda()

img2H, img2W = int(imgObject.shape[0]/2), int(imgObject.shape[1]/2)
# feats0 = extractor.extract(image0)  

countFrame = 0
countFrameFound = 0
PathReal = []
PathPred = []
PathVirtual = []
ListdetectResults = []
prezone = 10
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("C:/Users/TuanBao/Desktop/My_Docs/CNTT/imgMatching/lightGlue/video/plane1.mp4")
cap = cv2.VideoCapture('C:/Users/TuanBao/Desktop/My_Docs/CNTT/imgMatching/lightGlue/video/plane1.mp4')
while cap.isOpened():
    countFrame+=1
    start_time = time.time()
    ret, frame = cap.read()
    frame = frame[:800, :]
    # cv2.fillPoly(frame, pts=[np.array([[1020,400], [1400, 440], [1400, 600], [1020,560]])], color=(255,255,255))
    # frame = cv2.circle(frame, (1300,470), 40, (255,255,255),80)   #plane1 
    
 
    # frame = frame[550:1200, :]
    # frame = cv2.resize(frame,(1920,1080))
    # cv2.fillPoly(frame, pts=[np.array([[400,600], [900, 600], [900, 900], [400,900]])], color=(255,255,255))
    # frame = cv2.circle(frame, (700,600), 100, (255,255,255),160)    #plane2
    
    
    
    # frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
    image1 = numpy_image_to_torch(frame).cuda()
    # print(frame.shape)
    
    feats1 = extractor.extract(image1)
    # load the matcher
    feats0 = extractor.extract(image0)
   
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  

    # kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    
    m_kpts0_numpy = m_kpts1.cpu().numpy()
    m_kpts0_cv2 = m_kpts0_numpy.round().astype(int)
    
    KeyPoints = []
    X = []
    Y = []
    for keypoint in m_kpts0_cv2[:100]:
        # print(keypoint)
        KeyPoints.append([keypoint[0], keypoint[1]])
    # print(KeyPoints)
    if len(KeyPoints):
        data = np.array(KeyPoints)
        # print(data, data.shape)
        z_scores = np.abs(zscore(data, axis=0))


        threshold = 0.5
        filtered_data = data[(z_scores < threshold).all(axis=1)]
        print(len(filtered_data))
        if len(filtered_data) >= 3:
            for point in filtered_data:
                X.append(point[0])
                Y.append(point[1])
                frame = cv2.circle(frame, (point), 3, (255,255,2),-1)
    if len(X) and len(Y):
        countFrameFound+=1
        print('found')
        prezone = 10
        centerX = int(sum(X)/len(X))
        centerY = int(sum(Y)/len(Y))
        frame = cv2.rectangle(frame, (centerX-img2W, centerY+img2H) , (centerX+img2W, centerY-img2H), (0,255,0), 2)
        PathVirtual.append((centerX,centerY))
        print(PathVirtual[-1])
        print(img2W, img2H)
        detectResults = np.array([[float(centerX+img2W), float(centerY+img2H), float(centerX-img2W), float(centerY-img2H), float(0.6), 4 ]])
        detectResults = detectResults.astype(np.float64)
        ListdetectResults.append(detectResults)
        print('detectResults ',detectResults)
        update= tracker.update(detectResults, frame)
        LocationPred = tracker.predict_location(detectResults, frame)
        print('LocationPred', LocationPred)
        if LocationPred:
            # ratioList.append(int(LocationPred[0][2]))
            # print('LocationPred', LocationPred)
            # print(int(LocationPred[0][0]), int(LocationPred[0][1]))
            # im = cv2.line(im, (int(LocationPred[0][0]), int(LocationPred[0][1]+LocationPred[0][3]/2)), (int(LocationPred[0][0]), int(LocationPred[0][1])), (255,255,255), 1)
            # im = cv2.line(im, (int(LocationPred[0][0]+LocationPred[0][3]*LocationPred[0][2]/2), int(LocationPred[0][1])), (int(LocationPred[0][0]), int(LocationPred[0][1])), (255,255,255), 1)
            PathPred.append((int(LocationPred[0][0]), int(LocationPred[0][1])))
            print(PathPred[-1])

        
    else:
        print('not found')
        prezone+=7
        
        if len(ListdetectResults):
            # update= tracker.update(ListdetectResults[-1], frame)
            LocationVirtual = tracker.predict_location(ListdetectResults[-1], frame)
            print('LocationVirtual ', LocationVirtual)
            if LocationVirtual:
                PathVirtual.append((int(LocationVirtual[0][0]),int(LocationVirtual[0][1])))
                
                detection_resultsVirtual = np.array([[LocationVirtual[0][0]-img2W,LocationVirtual[0][1]-img2H, LocationVirtual[0][0]+img2W, LocationVirtual[0][1]+img2H,0.6, 4 ]])
                ListdetectResults.append(detection_resultsVirtual)
                # print('detection_resultsVirtual', detection_resultsVirtual)
                # print(LocationPred)
                # print(int(LocationPred[0][0]), int(LocationPred[0][1]))
                # im = cv2.line(im, (int(LocationPred[0][0]), int(LocationPred[0][1]+LocationPred[0][3]/2)), (int(LocationPred[0][0]), int(LocationPred[0][1])), (255,255,255), 1)
                # im = cv2.line(im, (int(LocationPred[0][0]+LocationPred[0][3]*LocationPred[0][2]/2), int(LocationPred[0][1])), (int(LocationPred[0][0]), int(LocationPred[0][1])), (255,255,255), 1)
                LocationPred = tracker.predict_location(detection_resultsVirtual, frame)
                print("location Pred: ",LocationPred)
                PathPred.append((int(LocationPred[0][0]), int(LocationPred[0][1])))
        

    # print(*PathVirtual)
    # print(*PathPred)
    for point in PathVirtual:
        frame = cv2.circle(frame, point, 2, (255,2,2),-1)
    for point in PathPred:     
        frame = cv2.circle(frame, point, 2, (2,2,255),-1)
    end_time = time.time()
    execution_time = end_time - start_time
    fps = 1/execution_time
    cv2.putText(frame,"FPS: "+str(int(fps)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('TEST lightglue', cv2.resize(frame, (1600,900)))
    print('{}/{}'.format(countFrameFound, countFrame))
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
cap.release()