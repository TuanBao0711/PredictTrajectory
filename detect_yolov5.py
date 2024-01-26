import os
import torch
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox

import torch.nn.functional as F
import time


class Detection:
    def __init__(self):
        # self.weights = os.path.join(ROOT, "/resources/Weight/face_v3.pt")
        self.imgsz = (320, 320)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = 'cuda'
        self.classes = None
        self.agnostic_nms = True
        self.half = False
        self.dnn = False  # use OpenCV DNN for ONNX inference

    def _load_model(self):
        # Load model
        # self.device = select_device(self.device)
        if self.device == "cpu":
            arg = "cpu"
        else:
            arg = f"{self.device}"
        self.device = torch.device(arg)
        self.model = DetectMultiBackend(
            self.weights, device=self.device, fp16=self.half)
        self.model.eval()
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check image size

    @torch.no_grad()
    def detect(self, image):
        bboxes = []
        im = letterbox(image, self.imgsz, stride=self.stride,
                       auto=self.pt)[0]  # resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        im = F.interpolate(im, size=(640, 640), mode='bilinear', align_corners=False)
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                   self.agnostic_nms, max_det=self.max_det)
        for i, det in enumerate(pred):
            if len(det):
                det = det.detach().cpu().numpy()
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = list(map(lambda x: max(0, int(x)), xyxy))
                    bboxes.append([x1, y1, x2, y2, int(cls), float(conf)])
        return bboxes


class Tracking(Detection):
    def __init__(self):
        super().__init__()
        self._tracker = Sort(max_age=150, min_hits=4, iou_threshold=0.1)

    @torch.no_grad()
    def track(self, image):
        track_dict = {}
        bboxes = self.detect(image)
        dets_to_sort = np.empty((0, 6))
        for x1, y1, x2, y2, cls, conf in bboxes:
            dets_to_sort = np.vstack(
                (dets_to_sort, np.array([x1, y1, x2, y2, conf, cls])))

        tracked_det = self._tracker.update(dets_to_sort)
        if len(tracked_det):
            bbox_xyxy = tracked_det[:, :4]
            indentities = tracked_det[:, 8]
            categories = tracked_det[:, 4]
            for i in range(len(bbox_xyxy)):
                x1, y1, x2, y2 = list(map(lambda x: max(0, int(x)), bbox_xyxy[i]))
                id_ = int(indentities[i])
                track_dict[id_] = (x1, y1, x2, y2, categories[i])
        return track_dict
