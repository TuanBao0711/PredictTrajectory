import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT
from boxmot import BYTETracker
from boxmot import StrongSORT
from ultralytics import YOLO

tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cuda:0',
    fp16=True,
)

# tracker = BYTETracker()
# tracker = StrongSORT(
#     model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
#     device='cuda:0',
#     fp16=True,
# )
vid = cv2.VideoCapture('C:/Users/TuanBao/Desktop/My_Docs/CNTT/imgMatching/lightGlue/video/plane1.mp4')
color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 0.5

Model = YOLO('model/yolov8m.pt')


while True:
    ret, im = vid.read()


    
    # im = im[150:1200, :]
    # im = cv2.resize(im,(1920,1080))
    # cv2.fillPoly(im, pts=[np.array([[600,600], [800, 600], [800, 900], [600,900]])], color=(255,255,255)) #obscured zone
    
    #plane1
    im = im[:750, :]
    cv2.fillPoly(im, pts=[np.array([[720,300], [1400, 340], [1400, 600], [720,560]])], color=(255,255,255))
    
    results = Model(im)
    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        cls = result.boxes.cls
        # convert PyTorch to NumPy
        boxes_np = boxes.cpu().numpy()
        confs_np = confs.cpu().numpy()
        cls_np = cls.cpu().numpy()
        detection_results = np.column_stack((boxes_np, confs_np, cls_np))
    tracks = tracker.update(detection_results, im) # --> (x, y, x, y, id, conf, cls, ind)
    # LocationPred = tracker.predict_location(detection_results, im)
    # print(LocationPred)
    if np.size(tracks) > 0:
        xyxys = tracks[:, 0:4].astype('int') # float64 to int
        ids = tracks[:, 4].astype('int') # float64 to int
        confs = tracks[:, 5]
        clss = tracks[:, 6].astype('int') # float64 to int
        inds = tracks[:, 7].astype('int') # float64 to int

        if tracks.shape[0] != 0:
            for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
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

    # show image with bboxes, ids, classes and confidences
    cv2.imshow('frame', im)

    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()