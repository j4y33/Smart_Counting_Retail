import re
import math
import numpy as np
import typing as typ
from io import BytesIO
from .structs import BoundingBox


import PIL.Image

try:
    import cv2
except ImportError:
    cv2 = None


def rgbcolor(h: int, f: float, v: float = 1.0, s: float = 1.0, p: float = 1.0):
    """Convert a color specified by h-value and f-value to an RGB
    three-tuple."""
    choices = {
        0: [v, f, p],
        1: [1 - f, v, p],
        2: [p, v, f],
        3: [p, 1 - f, v],
        4: [f, p, v],
        5: [v, p, 1 - f]
    }
    return list(map(lambda x: int(x * 255), choices.get(h, [v, f, p])))


def uniquecolors(n):
    """Compute a list of distinct colors, ecah of which is
    represented as an RGB three-tuple"""
    hues = [360.0 / n * i for i in range(n)]
    hs = (math.floor(hue / 60) % 6 for hue in hues)
    fs = (hue / 60 - math.floor(hue / 60) for hue in hues)
    return [rgbcolor(h, f) for h, f in zip(hs, fs)]


class ColorsIterator:
    def __init__(self, colors):
        self._colors = colors
        self._current = 0

    def get(self):
        self._current += 1
        if self._current >= len(self._colors):
            self._current = 0

        return self._colors[self._current]


class ColorPicker:

    def __init__(self, n_colors=20):
        self._colors = ColorsIterator(uniquecolors(n_colors))
        self._color_by_id = {}

    def get(self, idx):
        if idx not in self._color_by_id:
            self._color_by_id[idx] = self._colors.get()
        return self._color_by_id[idx]


def cosine_distance_vectorized(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Arguments:
        x: a numpy float array with shape [n, c].
        y: a numpy float array with shape [m, c].
    Returns:
        a numpy float array with shape [n, m].
    """
    epsilon = 1e-8

    x_norm = np.sqrt((x ** 2).sum(1, keepdims=True))
    y_norm = np.sqrt((y ** 2).sum(1, keepdims=True))

    x = x / (x_norm + epsilon)
    y = y / (y_norm + epsilon)

    product = np.expand_dims(x, 1) * y  # shape [n, m, c]
    cos = product.sum(2)  # shape [n, m]
    return 1.0 - cos


def cosine_distance(x: typ.List[np.ndarray], y: typ.List[np.ndarray]) -> np.ndarray:
    """
    cosine distance beetween [n, l, c] and [m, k, c]

    Parameters
    ----------
    x
        list with shape  [n, l, c]
    y
        list with shape  [m, k, c]

    Returns
    -------
        a numpy float array with shape [n, m].
    """
    num_x = len(x)
    num_y = len(y)
    cos_res = np.ones((num_x, num_y))
    for ind1 in range(num_x):
        for ind2 in range(num_y):
            # get cosine distance beetween 2 tracks
            cos_res[ind1, ind2] = cosine_distance_vectorized(x[ind1], y[ind2]).mean()
    return cos_res


def camel_to_snake_upper(name: str) -> str:
    """Convert `CamelCase` to `SNAKE_CASE` """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).upper()


def snake_to_camel(text: str) -> str:
    """Convert `SNAKE_CASE` to `CamelCase` """
    parts = text.lower().split('_')
    if len(parts) < 0:
        return text.title()
    return parts[0].title() + ''.join(p.title() for p in parts[1:])


def dict_to_lower_keys(dct: typ.Dict[str, typ.Any]) -> typ.Dict[str, typ.Any]:
    result = {}
    for k, v in dct.items():
        val = v
        if isinstance(v, dict):
            val = dict_to_lower_keys(v)
        result[k.lower()] = val
    return result


if cv2:

    def _resize(image: np.ndarray, new_size: typ.Tuple[int, int]) -> np.ndarray:
        return cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)

    def _compress(image: np.ndarray, quality: int = 90, extension: str = 'jpeg') -> bytes:
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

        result, enc_data = cv2.imencode('.' + extension.lower(), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not result:
            raise RuntimeError('Image compression error')

        return enc_data.tobytes()

    def _decompress(data: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError('Image decompression error')

        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        return image

else:

    def _resize(image: np.ndarray, new_size: typ.Tuple[int, int]) -> np.ndarray:
        return np.asarray(PIL.Image.fromarray(image).resize(new_size, resample=PIL.Image.NEAREST))

    def _compress(image: np.ndarray, quality: int = 90, extension: str = 'jpeg') -> bytes:
        img = PIL.Image.fromarray(image)
        with BytesIO() as buffer:
            img.save(buffer, extension.upper(), quality=quality)
            return buffer.getvalue()

    def _decompress(data: bytes) -> np.ndarray:
        img = np.asarray(PIL.Image.open(BytesIO(data)))
        if img.dtype == np.int32:
            img = img.astype(np.uint16)
        return img


def compress_image(image: np.ndarray, quality: int = 90, extension: str = 'jpeg') -> bytes:
    """Compress numpy array to JPEG or PNG image"""
    if extension.lower() not in {'jpeg', 'png'}:
        raise ValueError(
            'Unsupported extension "{}" for image compression'.format(extension))

    # GRAY16 can be compressed only in PNG format
    if len(image.shape) == 2 and image.dtype == np.uint16:
        extension = 'png'

    return _compress(image, quality, extension)


def decompress_image(data: bytes) -> np.ndarray:
    """Decompress JPEG or PNG image to numpy array"""
    return _decompress(data)


def resize(image: np.ndarray,
           size: typ.Tuple[int, int], aspect_ratio: bool = False) -> np.ndarray:
    """Resize single image
    :param image: RGB image as numpy array
    :param size: (width, height) tuple with new image size
    :param aspect_ratio: keep aspect ratio or not
    """
    origin_h, origin_w = image.shape[:2]
    if (origin_w, origin_h) == size:
        return image

    if aspect_ratio:
        scale_value = min(size[0] / origin_w, size[1] / origin_h)
        size = int(origin_w * scale_value), int(origin_h * scale_value)

    return _resize(image, size)


def resize_all(images: typ.List[np.ndarray],
               size: typ.Tuple[int, int], aspect_ratio: bool = False) -> typ.List[np.ndarray]:
    """Resize single image
    :param images: list wiht RGB images as numpy arrays
    :param size: (width, height) tuple with new image size
    :param aspect_ratio: keep aspect ratio or not
    """
    _expected_shape = (size[1], size[0])
    if all(img.shape[:2] == _expected_shape for img in images):
        return images

    if aspect_ratio:
        origin_h, origin_w = images[0].shape[:2]
        scale_value = min(size[0] / origin_w, size[1] / origin_h)
        size = int(origin_w * scale_value), int(origin_h * scale_value)

    return [_resize(image, size) for image in images]


def crop_by_rect(
        image: np.ndarray,
        rect: typ.Tuple[int, int, int, int],
        margin: typ.Optional[typ.Tuple[int, int]] = None) -> np.ndarray:
    """Crop image by rectangle
    :param image: source image to crop, shape [height, width, color] or [height, width] for grayscale/depth
    :param rect: crop rect [x_min, y_min, x_max, y_max]
    :param margin: additional margin borders [x_margin, y_margin]
    :return cropped image
    """
    x_min, y_min, x_max, y_max = rect
    img_height, img_width = image.shape[:2]

    if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
        raise ValueError('Wrong crop rectangle: {}'.format(rect))

    if x_min >= img_width or y_min >= img_height:
        raise ValueError('Crop position is out of the image')

    if margin is not None:
        x_margin, y_margin = margin
        if x_margin < 0 or y_margin < 0:
            raise ValueError('Crop margin can\'t be negative')
    else:
        x_margin, y_margin = 0, 0

    x_min, x_max = np.clip([x_min - x_margin, x_max + x_margin], 0, img_width)
    y_min, y_max = np.clip([y_min - y_margin, y_max + y_margin], 0, img_height)

    return image[y_min:y_max, x_min:x_max]


def crop_by_box(
        image: np.ndarray,
        bounding_box: typ.Tuple[int, int, int, int],
        margin: typ.Optional[typ.Tuple[int, int]] = None) -> np.ndarray:
    """Crop image by bounding box
    :param image: source image to crop, shape [height, width, color]
    :param bounding_box: crop box [x_min, y_min, width, height]
    :param margin: additional margin borders [x_margin, y_margin]
    :return cropped image
    """
    x_min, y_min, width, height = bounding_box
    return crop_by_rect(image, rect=(x_min, y_min, x_min + width, y_min + height), margin=margin)


def iou(boxes_a: typ.Union[np.ndarray, typ.List[BoundingBox]],
        boxes_b: typ.Union[np.ndarray, typ.List[BoundingBox]]) -> np.ndarray:
    """Computes pairwise intersection-over-union between two box collections.
    :param boxes_a: first box collection, shape [N, 4]
    :param boxes_b: second box collection, shape [N, 4]
    :return: a float numpy array with shape [N, M] representing pairwise iou scores
    """
    boxes_a = np.array(boxes_a, dtype=np.float32)
    boxes_b = np.array(boxes_b, dtype=np.float32)

    # to shapes like [None, 1]
    xmin_a, ymin_a, width_a, height_a = np.split(
        boxes_a, indices_or_sections=4, axis=1)
    xmin_b, ymin_b, width_b, height_b = np.split(
        boxes_b, indices_or_sections=4, axis=1)

    xmax_a, ymax_a = xmin_a + width_a, ymin_a + height_a
    xmax_b, ymax_b = xmin_b + width_b, ymin_b + height_b

    # intersect heights [N, K]
    all_pairs_min_ymax = np.minimum(ymax_a, np.transpose(ymax_b))
    all_pairs_max_ymin = np.maximum(ymin_a, np.transpose(ymin_b))
    intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)

    # intersect widths [N, K]
    all_pairs_min_xmax = np.minimum(xmax_a, np.transpose(xmax_b))
    all_pairs_max_xmin = np.maximum(xmin_a, np.transpose(xmin_b))
    intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)

    # intersection area
    intersection_area = intersect_widths * intersect_heights

    # union area
    area_box_a = (xmax_a - xmin_a) * (ymax_a - ymin_a)
    area_box_b = (xmax_b - xmin_b) * (ymax_b - ymin_b)

    union_area = (area_box_a + np.transpose(area_box_b) - intersection_area)

    # limits to [0, 1]
    return np.clip(intersection_area / np.maximum(union_area, 1e-8), 0.0, 1.0)
