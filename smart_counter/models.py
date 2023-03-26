import numpy as np
import trafaret as t
import typing as typ

from .error import ApplicationError
from .base import Model
from .structs import KeyPoint, BoundingBox
from .structs import Detection

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from torchvision.transforms import Normalize, ToTensor, Resize, Compose
import torch
from torchreid import models
from torchreid.utils import FeatureExtractor
from PIL import Image
from collections import OrderedDict


class PersonDetectron2(Model):

    _config_schema = t.Dict({
        t.Key('weights'): t.String(min_length=3),
        t.Key('detector_threshold', optional=True, default=.3): t.Float(gte=0, lte=1.0),
        t.Key('keypoints_threshold', optional=True, default=.05): t.Float(gte=0, lte=1.0),
    }, ignore_extra='*')

    # keypoint names for COCO format
    keypoints_names_map = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }

    def on_startup(self):
        # get base configs from model zoo
        cfg = get_cfg()
        cfg.merge_from_file(self.config['weights'])
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config['weights'])
        # set threshold
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.config['detector_threshold']
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config['detector_threshold']
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.config[
            'detector_threshold']
        # get predictor
        try:
            self._model = DefaultPredictor(cfg)
        except Exception:
            raise ApplicationError(f'error while loading model weights from file for {self}')

    def on_shutdown(self):
        if hasattr(self, '_model'):
            self._model = None

    def process_single(self, image: np.ndarray, **extra) -> typ.List[Detection]:
        """
        process single image with detectron model

        Parameters
        ----------
        image
            image to process

        Returns
        -------
            array of person detections for this image
        """
        detections = []
        person_class = 0
        predictions = self._model(image)
        instances = predictions['instances'][predictions['instances'].pred_classes == person_class]

        for candidate in range(len(instances)):
            box = list(
                map(int, instances[candidate].pred_boxes.tensor.cpu().numpy()[0]))
            # wigth and heigth
            box[2] -= box[0]
            box[3] -= box[1]
            confidence = instances[candidate].scores[0].cpu().numpy()
            keypoints = list(
                map(lambda x: x.tolist(), instances[candidate].pred_keypoints.cpu().numpy()))
            keypoints = {self.keypoints_names_map[ind]: KeyPoint(*[int(kp[0]), int(kp[1]), kp[2]])
                         for ind, kp in enumerate(keypoints[0])
                         if kp[2] > self.config['keypoints_threshold']}
            detection = {
                'box': BoundingBox(*box),    # typ.Tuple[int, int, int, int]
                'score': float(confidence),  # float
                'keypoints': keypoints       # typ.Dict[str:KeyPoint]
            }
            detections.append(detection)
        return detections


class PersonEmbeddingTorchReid(Model):
    # params for model
    _NUM_CLASSES = 4101

    # params for transform
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]
    _INPUT_SHAPE = (256, 128)

    _config_schema = t.Dict({
        t.Key('weights', optional=False): t.String(min_length=3),
        t.Key('device', optional=True, default='cuda:0'): t.String(min_length=3),
        t.Key('model_name', optional=True, default='osnet_ain_x1_0'): t.String(min_length=3)
    }, ignore_extra='*')

    def on_startup(self):
        if self.config['weights'].endswith('.pth'):
            s = torch.load(self.config['weights'])
            s = OrderedDict([(k[7:], v) for k, v in s.items()])
            model = models.osnet_ain.osnet_ain_x1_0(
                num_classes=self._NUM_CLASSES, pretrained=False)
            model.load_state_dict(s)
            model.fc[1] = torch.nn.Identity()
            model.fc[2] = torch.nn.Identity()
            model.classifier = torch.nn.Identity()
            self._model = model.eval().to(self.config['device'])
        else:
            self._model = FeatureExtractor(
                model_name=self.config['model_name'],
                model_path=self.config['weights'],
                device=self.config['device']
            )
        self._transform = Compose([
            Resize(self._INPUT_SHAPE),
            ToTensor(),
            Normalize(mean=self._MEAN, std=self._STD),
        ])

    def on_shutdown(self):
        if hasattr(self, '_model'):
            self._model = None

    def process_single(self, image: np.ndarray, **extra) -> np.ndarray:
        """
        process single image with torchreid feature extractor

        Parameters
        ----------
        image
            image to process

        Returns
        -------
            list of shape (N) with embeddings
        """
        with torch.no_grad():
            x = self._transform(Image.fromarray(image))
            x = x.unsqueeze(0).to(self.config['device'])
            embeddings = self._model(x).cpu().numpy()
            # embeddings shape is (1, N), so return (N) as we always pass only 1 frame to this method
            return embeddings[0]
