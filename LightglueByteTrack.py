import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT
from boxmot import BYTETracker
from boxmot import StrongSORT

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
from scipy.stats import zscore

# tracker = DeepOCSORT(
#     model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
#     device='cuda:0',
#     fp16=True,
# )

tracker = BYTETracker()
# tracker = StrongSORT(
#     model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
#     device='cuda:0',
#     fp16=True,
# )
vid = cv2.VideoCapture('C:/Users/TuanBao/Desktop/My_Docs/CNTT/imgMatching/lightGlue/video/plane1.mp4')
color = (0, 255, 2)  # BGR
thickness = 2
fontscale = 0.5


PathReal = []
PathVitual = []
PathPred = []
DetectList = []

ratioList = []
Pause = False
predZone = 50


extractor = SIFT(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='sift').eval().cuda()


imgObject = cv2.imread("C:/Users/TuanBao/Desktop/My_Docs/CNTT/imgMatching/lightGlue/mig/plane1.png")
# imgObject = cv2.fastNlMeansDenoisingColored(imgObject,None,10,10,7,21)
imgObject = cv2.cvtColor(imgObject, cv2.COLOR_BGR2GRAY)
# cv2.imshow('object',imgObject)
image0 = numpy_image_to_torch(imgObject).cuda()

img2H, img2W = int(imgObject.shape[0]/2), int(imgObject.shape[1]/2)

def modelLightglue(image0, frame):
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
        if len(filtered_data) >= 3:
            for point in filtered_data:
                X.append(point[0])
                Y.append(point[1])
            centerX = int(sum(X)/len(X))
            centerY = int(sum(Y)/len(Y))
            frame = cv2.rectangle(frame, (centerX-img2W, centerY+img2H) , (centerX+img2W, centerY-img2H), (0,255,0), 2)
            detectResults = np.array([[float(centerX+img2W), float(centerY+img2H), float(centerX-img2W), float(centerY-img2H), float(0.65), 4 ]])
            detectResults = detectResults.astype(np.float64)
            return detectResults
        
    
    
LocationPredGlobal = np.array([])
while True:
    if not Pause:
        ret, im = vid.read()

        #plane1
        im = im[:750, :]
        cv2.fillPoly(im, pts=[np.array([[720,300], [1400, 340], [1400, 600], [720,560]])], color=(255,255,255))

        detection_results = modelLightglue(image0 , im)
        # print('detect results: ', detection_results)
        # print(len(detection_results))
        
        if detection_results is not None:
                print('Found')

                DetectList.append(detection_results)
                print('detection_results', detection_results)
      

                tracks = tracker.update(detection_results, im) # --> (x, y, x, y, id, conf, cls, ind)
                LocationPred = tracker.predict_location(detection_results, im)
                
                if LocationPred:
                    ratioList.append(int(LocationPred[0][2]))
                    
                    PathPred.append((int(LocationPred[0][0]), int(LocationPred[0][1])))
                    LocationPredGlobal = LocationPred
                    
                else:
                    LocationPred = LocationPredGlobal
                print('LocationPred', LocationPred)
                if np.size(tracks) > 0:
                    xyxys = tracks[:, 0:4].astype('int') # float64 to int
                    ids = tracks[:, 4].astype('int') # float64 to int
                    confs = tracks[:, 5]
                    clss = tracks[:, 6].astype('int') # float64 to int
                    inds = tracks[:, 7].astype('int') # float64 to int

                    if tracks.shape[0] != 0:
                        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                            if (cls == 4):
                                im = cv2.rectangle(
                                    im,
                                    (xyxy[0], xyxy[1]),
                                    (xyxy[2], xyxy[3]),
                                    color,
                                    thickness
                                )
                                cv2.putText(
                                    im,
                                    f'id: {id}, conf: {conf}, c: {cls}',
                                    (xyxy[0], xyxy[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    fontscale,
                                    color,
                                    thickness
                                )
                                
                                PathReal.append((int((xyxy[0]+ xyxy[2])/2),int((xyxy[1]+ xyxy[3])/2)))
                                PathVitual.append((int((xyxy[0]+ xyxy[2])/2),int((xyxy[1]+ xyxy[3])/2)))
                # for point in PathReal:
                #     im = cv2.circle(im, point, 2, (255,2,2),-1)
                # for point in PathPred:
                
                #     im = cv2.circle(im, point, 2, (2,2,255),-1)
                predZone = 50
            


        else:
            print('not Found')
            if len(DetectList):
                detection_results = DetectList[-1]
                print('detection_results', detection_results)

                LocationVirtual = tracker.predict_location(detection_results, im)
                print('LocationVirtual ', LocationVirtual)
                if LocationVirtual:
                    PathVitual.append((int(LocationVirtual[0][0]), int(LocationVirtual[0][1])))
                    print(LocationVirtual)
                    detection_resultsVitual = np.array([[LocationVirtual[0][0]-LocationVirtual[0][3]*LocationVirtual[0][2]/2,LocationVirtual[0][1]-LocationVirtual[0][3]/2, LocationVirtual[0][0]+LocationVirtual[0][3]*LocationVirtual[0][2]/2, LocationVirtual[0][1]+LocationVirtual[0][3]/2,detection_results[0][-2], detection_results[0][-1]] ])
                    DetectList.append(detection_resultsVitual)
                    LocationPred = tracker.predict_location(detection_resultsVitual, im)
                    print("location Pred: ",LocationPred)
                    PathPred.append((int(LocationPred[0][0]), int(LocationPred[0][1])))
                im = cv2.circle(im, (int(LocationPred[0][0]), int(LocationPred[0][1])), predZone, (2,2,255),1)
        for point in PathVitual:
            im = cv2.circle(im, point, 2, (255,2,2),-1)
        for point in PathPred:
        
            im = cv2.circle(im, point, 2, (2,2,255),-1)
        predZone +=6
        
    # show image with bboxes, ids, classes and confidences
    
        cv2.imshow('frame', cv2.resize(im,(1600,900)))
    key = cv2.waitKey(1) & 0xFF

    # Nếu phím cách được bấm, thay đổi trạng thái pause/tiếp tục
    if key == ord(' '):
        Pause = not Pause
    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()