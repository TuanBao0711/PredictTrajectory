import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT
from boxmot import BYTETracker
from boxmot import StrongSORT
from ultralytics import YOLO

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

Model = YOLO('model/yolov8m.pt')

PathReal = []
PathVitual = []
PathPred = []
DetectList = []

ratioList = []
Pause = False
predZone = 50
while True:
    if not Pause:
        ret, im = vid.read()

        #plane1
        im = im[:750, :]
        cv2.fillPoly(im, pts=[np.array([[720,300], [1400, 340], [1400, 600], [720,560]])], color=(255,255,255))
        # im = cv2.circle(im, (1300,470), 40, (255,255,255),80)   
    
        #plane2
        # im = im[150:1200, :]
        # im = cv2.resize(im,(1920,1080))
        # cv2.fillPoly(im, pts=[np.array([[400,600], [900, 600], [900, 900], [400,900]])], color=(255,255,255)) #obscured zone
        # cv2.fillPoly(im, pts=[np.array([[600,600], [1100, 600], [1500, 900], [1000,900]])], color=(255,255,255)) #obscured zone
        # # cv2.circle(im, (800,750), 50, (255,255,255),140)
        # # cv2.circle(im, (900,750), 50, (255,255,255),140)
        
        # im = cv2.circle(im, (1000,470), 140, (255,255,255),180)   
        
        results = Model(im)
        
        if results[0].boxes.data.tolist():
            detectResult = []
            for result in results:
                clsDetect = result.boxes.cls
                clsDetect_np =clsDetect.cpu().numpy()
                detectResult.append(clsDetect_np[0])
            # print('detectResult ',detectResult)
            if 4.0 in detectResult:
                print('Found')
                for result in results:
                    boxes = result.boxes.xyxy
                    
                    confs = result.boxes.conf
                    cls = result.boxes.cls
                    # convert PyTorch to NumPy
                    boxes_np = boxes.cpu().numpy()
                    # print('xyxy np ', boxes_np)
                    # print('type xyxy np: ', type(boxes_np))
                    confs_np = confs.cpu().numpy()
                    cls_np = cls.cpu().numpy()
                    if (cls_np[0]==4):
                        detection_results = np.column_stack((boxes_np, confs_np, cls_np))
                DetectList.append(detection_results)
                print('detection_results', detection_results)
      
                # im =  cv2.circle(im, (int(detection_results[0][0]),int(detection_results[0][1])), 2, (255,2,255),1)
                # im =  cv2.circle(im, (int(detection_results[0][2]),int(detection_results[0][3])),2, (255,255,2),1)
                # im =  cv2.circle(im, int(detection_results[0][2]), 2, (255,255,255),1)
                # im =  cv2.circle(im, int(detection_results[0][3]), 2, (2,255,255),1)
                tracks = tracker.update(detection_results, im) # --> (x, y, x, y, id, conf, cls, ind)
                LocationPred = tracker.predict_location(detection_results, im)
                if LocationPred:
                    ratioList.append(int(LocationPred[0][2]))
                    print('LocationPred', LocationPred)
                    # print(int(LocationPred[0][0]), int(LocationPred[0][1]))
                    # im = cv2.line(im, (int(LocationPred[0][0]), int(LocationPred[0][1]+LocationPred[0][3]/2)), (int(LocationPred[0][0]), int(LocationPred[0][1])), (255,255,255), 1)
                    # im = cv2.line(im, (int(LocationPred[0][0]+LocationPred[0][3]*LocationPred[0][2]/2), int(LocationPred[0][1])), (int(LocationPred[0][0]), int(LocationPred[0][1])), (255,255,255), 1)
                    PathPred.append((int(LocationPred[0][0]), int(LocationPred[0][1])))
                    
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
                detection_results = DetectList[-1]
                # print('detection_results', detection_results)

                LocationVirtual = tracker.predict_location(detection_results, im)
                if LocationVirtual:
                    PathVitual.append((int(LocationVirtual[0][0]), int(LocationVirtual[0][1])))
                    print(LocationVirtual)
                    # print('LocationVirtual[0][0]-LocationVirtual[0][3]*LocationVirtual[0][2]/2 ', LocationVirtual[0][0]-LocationVirtual[0][3]*LocationVirtual[0][2]/2)
                    # print('LocationVirtual[0][1]-LocationVirtual[0][3]/2 ', LocationVirtual[0][1]-LocationVirtual[0][3]/2)
                    # print(' LocationVirtual[0][0]+LocationVirtual[0][3]*LocationVirtual[0][2]/2',  LocationVirtual[0][0]+LocationVirtual[0][3]*LocationVirtual[0][2]/2)
                    # print('LocationVirtual[0][1]+LocationVirtual[0][3]/2', LocationVirtual[0][1]+LocationVirtual[0][3]/2)
                    # print('detection_results[0][-2] - conf ', detection_results[0][-2])
                    # print('detection_results[0][-1] - cls ',detection_results[0][-1])
                    
                    detection_resultsVitual = np.array([[LocationVirtual[0][0]-LocationVirtual[0][3]*LocationVirtual[0][2]/2,LocationVirtual[0][1]-LocationVirtual[0][3]/2, LocationVirtual[0][0]+LocationVirtual[0][3]*LocationVirtual[0][2]/2, LocationVirtual[0][1]+LocationVirtual[0][3]/2,detection_results[0][-2], detection_results[0][-1]] ])
                    # print('detection_resultsVitual ',detection_resultsVitual)
                    DetectList.append(detection_resultsVitual)
                    # print('detection_resultsVitual', detection_resultsVitual)
                    # print(LocationPred)
                    # print(int(LocationPred[0][0]), int(LocationPred[0][1]))
                    # im = cv2.line(im, (int(LocationPred[0][0]), int(LocationPred[0][1]+LocationPred[0][3]/2)), (int(LocationPred[0][0]), int(LocationPred[0][1])), (255,255,255), 1)
                    # im = cv2.line(im, (int(LocationPred[0][0]+LocationPred[0][3]*LocationPred[0][2]/2), int(LocationPred[0][1])), (int(LocationPred[0][0]), int(LocationPred[0][1])), (255,255,255), 1)
                    LocationPred = tracker.predict_location(detection_resultsVitual, im)
                    # print(LocationPred)
                    PathPred.append((int(LocationPred[0][0]), int(LocationPred[0][1])))
                        # PathVitual.append((int(LocationPred[0][0]), int(LocationPred[0][1])))
                    
                        # for point in PathVitual:
                        #     im = cv2.circle(im, point, 2, (255,2,2),-1)
                        # for point in PathPred:
                        
                        #     im = cv2.circle(im, point, 2, (2,2,255),-1)
                        # predZone +=7
                        # im = cv2.circle(im, (int(LocationPred[0][0]), int(LocationPred[0][1])), predZone, (2,2,255),1)
        
                im = cv2.circle(im, (int(LocationPred[0][0]), int(LocationPred[0][1])), predZone, (2,2,255),1)

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
                    # print('LocationVirtual[0][0]-LocationVirtual[0][3]*LocationVirtual[0][2]/2 ', LocationVirtual[0][0]-LocationVirtual[0][3]*LocationVirtual[0][2]/2)
                    # print('LocationVirtual[0][1]-LocationVirtual[0][3]/2 ', LocationVirtual[0][1]-LocationVirtual[0][3]/2)
                    # print(' LocationVirtual[0][0]+LocationVirtual[0][3]*LocationVirtual[0][2]/2',  LocationVirtual[0][0]+LocationVirtual[0][3]*LocationVirtual[0][2]/2)
                    # print('LocationVirtual[0][1]+LocationVirtual[0][3]/2', LocationVirtual[0][1]+LocationVirtual[0][3]/2)
                    # print('detection_results[0][-2] - conf ', detection_results[0][-2])
                    # print('detection_results[0][-1] - cls ',detection_results[0][-1])
                    
                    detection_resultsVitual = np.array([[LocationVirtual[0][0]-LocationVirtual[0][3]*LocationVirtual[0][2]/2,LocationVirtual[0][1]-LocationVirtual[0][3]/2, LocationVirtual[0][0]+LocationVirtual[0][3]*LocationVirtual[0][2]/2, LocationVirtual[0][1]+LocationVirtual[0][3]/2,detection_results[0][-2], detection_results[0][-1]] ])
                    DetectList.append(detection_resultsVitual)
                    # print('detection_resultsVitual', detection_resultsVitual)
                    # print(LocationPred)
                    # print(int(LocationPred[0][0]), int(LocationPred[0][1]))
                    # im = cv2.line(im, (int(LocationPred[0][0]), int(LocationPred[0][1]+LocationPred[0][3]/2)), (int(LocationPred[0][0]), int(LocationPred[0][1])), (255,255,255), 1)
                    # im = cv2.line(im, (int(LocationPred[0][0]+LocationPred[0][3]*LocationPred[0][2]/2), int(LocationPred[0][1])), (int(LocationPred[0][0]), int(LocationPred[0][1])), (255,255,255), 1)
                    LocationPred = tracker.predict_location(detection_resultsVitual, im)
                    print("location Pred: ",LocationPred)
                    PathPred.append((int(LocationPred[0][0]), int(LocationPred[0][1])))
                        # PathVitual.append((int(LocationPred[0][0]), int(LocationPred[0][1])))
                    
                        # for point in PathVitual:
                        #     im = cv2.circle(im, point, 2, (255,2,2),-1)
                        # for point in PathPred:
                        
                        #     im = cv2.circle(im, point, 2, (2,2,255),-1)
                        # predZone +=7
                        # im = cv2.circle(im, (int(LocationPred[0][0]), int(LocationPred[0][1])), predZone, (2,2,255),1)
        
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