import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import torch 
from ultralytics.data.augment import LetterBox
from triton import TritonRemoteModel
from ultralytics.utils import ops
from ultralytics.engine.results import Results


import time

class CustomProfile:
    def __init__(self, description=""):
        self.description = description
        self.start_time = None
        self.elapsed_time = 0.0

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        print(f"{self.description}: {self.elapsed_time:.3f} seconds")

def pre_transform(im):
    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox(640, stride=32)
    return [letterbox(image=x) for x in im]


def preprocess_image(image_path, img_size=640):
    with CustomProfile("Preprocessing"):
        image = cv2.imread(image_path)
        not_tensor = not isinstance(image, torch.Tensor)
        if not_tensor:
            im = np.stack(pre_transform([image]))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # Contiguous
            im = torch.from_numpy(im)
        im = im.to('cpu')
        im = im.float()
        if not_tensor:
            im /= 255  # Normalize
        return im, image


def postprocess(preds, img, orig_img):
    with CustomProfile("Postprocessing"):
        preds = ops.non_max_suppression(
            preds,
            0.25,  # Confidence threshold
            0.45,  # Intersection over Union threshold
            classes=[0,5]   # TODO here for person and bus
        )
        preds = np.array(preds)
        preds[:, :4] = ops.scale_boxes(img.shape[2:], preds[:, :4], orig_img.shape)
        pred = torch.tensor(preds[0])
        results = [Results(orig_img, names=names, boxes=pred, path='./runs')]
        return results


def infer_with_triton(image_path):
    model = TritonRemoteModel('grpc://localhost:8001/yolo11m')

    # Preprocess
    input_image, original_image = preprocess_image(image_path)

    # Inference
    with CustomProfile("Inference"):
        output_data = model(input_image.cpu().numpy())

    output_data = torch.tensor(output_data[0])

    # Postprocess
    results = postprocess(output_data, input_image, original_image)

    # Print the results
    print("Detected bounding boxes (scaled to original image):")
    for res in results:
        print(res.boxes)
        print(res.verbose())


names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
    12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
    26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
    33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


# Example usage
image_path = '/home/touti/dev/triton_manager/triton_examples/bus.jpg'
infer_with_triton(image_path)
