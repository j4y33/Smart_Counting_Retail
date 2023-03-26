import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import dask.bag as db
import trafaret as t
from shapely.geometry import Polygon, Point
from pandas import DataFrame
from isodate import parse_datetime, parse_time
from typing import Union, NamedTuple


class TracksDrawer:
    _BOX_THICKNESS = 8
    _BOX_FONT = ImageFont.load_default()
    _LINE_THICKNESS = 10
    _CIRCLE_RADIUS = 30

    _EVENT_ZONE_COLOR = (0, 255, 0, 130)
    _FILTER_ZONE_COLOR = (255, 0, 0, 130)

    def __init__(self, background: Union[Image.Image, str]):
        """
        Class to draw tracks on shop screenshot
        :param background: path to the background picture or the picture in ndarray format
        """
        if isinstance(background, Image.Image):
            self.background = background
        elif type(background) == str:
            self.background = Image.open(background.split('/')[-1])
        else:
            self.background = Image.open(background)

        self.background.convert('RGBA')

    def draw(self, tracks: db.Bag, *, step: int = 1, location='head',
             show_arrows: bool = False, show_boxes: bool = False,
             filter_zone: Polygon = None, event_zone: Polygon = None) -> Image.Image:
        """
        Draws tracks on a background image
        :param tracks: tracks to be drawn
        :param step: iteration step
        :param location: location of keypoint to draw arrows
        :param show_arrows: True to draw tracks as arrows
        :param show_boxes: True to draw tracks as boxes
        :param filter_zone: shapely Polygon, None if no zone drawing needed
        :param event_zone: shapely Polygon, None if no zone drawing needed
        :return: PIL Image
        """

        def draw_box(frame: Image.Image, box: tuple, track_id: str, color: tuple) -> Image.Image:
            """
            :param frame: image to draw on
            :param box: bounding box in (x_min, y_min, x_max, y_max) format
            :param track_id: anything convertible to string
            :param color: tuple in (R, G, B, (A)) format
            :return: image with draws bounding boxes
            """
            x1, y1, x2, y2 = box
            text = str(track_id)
            text_width, text_height = self._BOX_FONT.getsize(text)

            bbox_draw = ImageDraw.Draw(frame)
            bbox_draw.rectangle(box, width=self._BOX_THICKNESS, outline=color)
            bbox_draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill=color)
            bbox_draw.text([x1, y1 - text_height], text, font=self._BOX_FONT)
            return frame

        def draw_line(frame: Image.Image, box: tuple, box_prev: tuple, color: tuple,
                      location: str = 'head') -> Image.Image:
            """
            Draws a line from @box_prev to @box
            :param frame: Image to draw on
            :param box: current box in (x_min, y_min, x_max, y_max) format
            :param box_prev: previous box in (x_min, y_min, x_max, y_max) format
            :param color: tuple in (R, G, B, A) format
            :param location: what keypoint to draw ('head'/'center')
            :return: PIL Image
            """
            x1, y1, x2, y2 = box
            p_x1, p_y1, p_x2, p_y2 = box_prev
            line_drawer = ImageDraw.Draw(frame)
            pt1_x = (x1 + x2) // 2
            pt2_x = (p_x1 + p_x2) // 2
            pt1_y = y1 if location == 'head' else (y1 + y2) // 2
            pt2_y = p_y1 if location == 'head' else (p_y1 + p_y2) // 2
            line_drawer.line([(pt1_x, pt1_y), (pt2_x, pt2_y)], fill=color, width=self._LINE_THICKNESS)
            return frame

        def draw_circle(frame: Image.Image, box: tuple, color: tuple, location: str = 'head') -> Image.Image:
            """
            Draws a circle of a detection's @box
            :param frame: Image to draw on
            :param box: bounding box in (x_min, y_min, x_max, y_max) format
            :param color: tuple in (R, G, B, (A)) format
            :param location: what keypoint to draw ('head'/'center')
            :return: PIL Image
            """
            x1, y1, x2, y2 = box
            circle_draw = ImageDraw.Draw(frame)
            pt_x = (x1 + x2) // 2
            pt_y = y1 if location == 'head' else (y1 + y2) // 2
            circle_draw.ellipse([(pt_x - self._CIRCLE_RADIUS // 2, pt_y - self._CIRCLE_RADIUS // 2),
                                 (pt_x + self._CIRCLE_RADIUS // 2, pt_y + self._CIRCLE_RADIUS // 2)],
                                fill=color, outline=color)
            return frame

        def draw_polygon(frame: Image.Image, points: list, color: tuple) -> Image.Image:
            """
            Draws polygon on @frame
            :param frame: frame to draw in
            :param points: list of points
            :param color: tuple in (R, G, B, A) format
            :return: PIL Image
            """
            poly = Image.new('RGBA', frame.size)
            poly_draw = ImageDraw.Draw(poly)
            poly_draw.polygon(points, fill=color, outline=color)
            frame.paste(poly, mask=poly)
            return frame

        pic = self.background.copy()
        for track in tracks:
            trajectory = track['trajectory']
            color = tuple(np.random.randint(0, 255, 3))
            for i in range(0, len(trajectory), step):
                if show_arrows:
                    if i != 0:
                        pic = draw_line(pic, trajectory[i]['bounding_box'], trajectory[i - 1]['bounding_box'],
                                        color=color, location=location)
                        pic = draw_circle(pic, trajectory[i]['bounding_box'], color=color, location=location)
                    elif len(trajectory) + 1 < step:
                        pic = draw_circle(pic, trajectory[i]['bounding_box'], color=color, location=location)

                if show_boxes:
                    pic = draw_box(pic, trajectory[i]['bounding_box'], track['track_id'], color)
        if type(filter_zone) == list:
            pic = draw_polygon(pic, filter_zone, self._FILTER_ZONE_COLOR)
        elif type(filter_zone) == Polygon:
            pic = draw_polygon(pic, filter_zone.exterior.coords, self._FILTER_ZONE_COLOR)

        if type(event_zone) == list:
            pic = draw_polygon(pic, event_zone, self._EVENT_ZONE_COLOR)
        elif type(event_zone) == Polygon:
            pic = draw_polygon(pic, event_zone.exterior.coords, self._EVENT_ZONE_COLOR)
        return pic


class TrackStats:
    @staticmethod
    def std(track: list, *, axis: int = 1) -> np.float64:
        """
        Calculates standard deviation of some track's coordinate, depending on @axis
        :param track: track, i.e. list of detections
        :param axis: axis of track, 1 for min_y coordinate, 0 for min_x
        :return: standard deviation value
        """
        return np.std([detection['bounding_box'][axis] for detection in track])

    @staticmethod
    def mean(track: list, *, axis: int) -> np.float64:
        """
        Calculates track's mean width/height, depending on @axis
        :param track: track, i.e. list of detections
        :param axis: 1 for height, 0 for width
        :return: standard deviation value
        """
        return np.mean([detection['bounding_box'][axis + 2] - detection['bounding_box'][axis] for detection in track])

    @staticmethod
    def in_zone(track: list, zone: Polygon, *, index: int = None, min_occurrences: int = 0) -> bool:
        """
        Checks whether the middle of detection with index @index (or every detection) of track is inside @zone
        :param track: track, i.e. list of detections
        :param zone: shapely Polygon
        :param index: index of detection to be checked. 0 for the first one, 1 for the last, None for every detection
        :param min_occurrences: check when track's detections present in zone at least for @min_occurrences
                               if (None, 0) -> all detections to be present in zone
        :return: True if track is within zone, False if not
        """

        def det_in_zone(detection: tuple) -> bool:  # Helper function to check single detection
            return zone.contains(Point(detection[0] + detection[2] / 2,
                                       detection[1] + detection[3] / 2))

        if index is None:
            num_dets_in_zone = np.sum([det_in_zone(detection['bounding_box']) for detection in track])
            if isinstance(min_occurrences, int) and min_occurrences > 0:
                return num_dets_in_zone >= min_occurrences
            return num_dets_in_zone >= len(track)

        return det_in_zone(track[index]['bounding_box'])


class TracksFilter:
    """
    Class to filter tracks
    Parameters:
        min_num_det: int
                - minimum number of detections in track
        min_y_std: int
                - minimum standard deviation of track's min_y coordinate
        min_size: tuple
                - (min_W, min_H) - which stands for minimum median bounding box width/height respectively
        event_zone: list(tuple)
                - zone to gather events from, format: [(x1, y1), (x2, y2),...], i.e. list of points
        filter_zone: list(tuple)
                - zone for 'only in zone filter', format: [(x1, y1), (x2, y2),...], i.e. list of points
        use_zone_filter: bool
                - True to use zone filtering, False if no filtering needed
        start_time: str
                - lower time bound to filter tracks in "HH:MM:SS" format (e.g. "12:34")
        end_time: str
                - upper time bound tracks in "HH:MM:SS" format (e.g. "12:34")

    Usage example:
        >>>
        ... tracks = TracksFilter('tracks.json', params={'min_num_det': 50}).filter()
        ... cv2.imwrite('result.png', TracksDrawer('background.png').draw(tracks.compute(),
        ... filter_zone=tracks.params['filter_zone'], event_zone=tracks.params['event_zone']))
        >>>
    """
    # Config to check types
    _config_schema = t.Dict({
        t.Key('min_num_det', optional=True, default=1): t.Int(),
        t.Key('min_y_std', optional=True, default=0): t.Int(),
        t.Key('min_occurrences', optional=True, default=0): t.Int(),
        t.Key('min_size', optional=True, default=(0, 0)):
            t.Tuple(t.Int, t.Int) >> (lambda x: NamedTuple('pair', [('w', int), ('h', int)])(*x)),
        t.Key('event_zone', optional=True, default=[(700, 0), (1270, 0), (1270, 370), (700, 370)]):
            t.List(t.Tuple(t.Int, t.Int)) >> (lambda l: Polygon(l)),
        t.Key('filter_zone', optional=True, default=[(700, 0), (1270, 0), (1270, 370), (700, 370)]):
            t.List(t.Tuple(t.Int, t.Int)) >> (lambda l: Polygon(l)),
        t.Key('use_zone_filter', optional=True, default=False): t.Bool,
        t.Key('start_time', optional=True, default='00:00:00'): t.Regexp(r'\d{2}:\d{2}:\d{2}') >> parse_time,
        t.Key('end_time', optional=True, default='23:59:59'): t.Regexp(r'\d{2}:\d{2}:\d{2}') >> parse_time,
    }, ignore_extra='*')

    def __init__(self, tracks: Union[db.Bag, str], *,
                 params: Union[dict, str] = {},
                 picture_size=None) -> None:
        """
        :param tracks: dask.DataFrame with tracks, or path to json file
        :param params: dictionary with parameters or path to json with parameters
        :param picture_size: size of the target picture (to rescale coordinates)
        """
        if type(tracks) == str:
            self.tracks = db.read_text(tracks).map(json.loads)
        elif type(tracks) == db.Bag:
            self.tracks = tracks
        else:
            raise ValueError("Illegal value for 'tracks'")

        video_info = self.tracks.take(1)[0]['video']
        self.frame_size = (video_info['frame_width'], video_info['frame_heigth'])

        if picture_size is not None:
            self.rescale_tracks(picture_size)

        self.set_params(params)

    def rescale_tracks(self, pic_size):
        """
        Rescales all the coordinates in tracks to suit @picture_size
        :param pic_size: size of the picture in (w, h) format
        :return:
        """

        def rescale_bbox(track):
            track = track.copy()
            for i in range(len(track['trajectory'])):
                track['trajectory'][i]['bounding_box'] = [coord * (w if i % 2 == 0 else h) for i, coord in
                                                          enumerate(track['trajectory'][i]['bounding_box'])]
            return track

        if self.frame_size == pic_size:
            return self

        w = pic_size[0] / self.frame_size[0]  # width factor
        h = pic_size[1] / self.frame_size[1]  # height factor

        self.tracks = self.tracks.map(rescale_bbox)
        self.frame_size = pic_size
        return self

    def set_params(self, params: Union[dict, str]) -> 'TracksFilter':
        """
        Sets parameters to filter tracks
        :param params: dictionary or path to json file with parameters. Be careful with the format!
        """
        if type(params) == str:
            with open(params, 'r') as f:
                params_dict = json.load(f)
        elif type(params) == dict:
            params_dict = params
        else:
            raise ValueError("Invalid value for 'params'")

        self.params = self._config_schema.check(params_dict)

        return self

    def export_params(self, file_name: str) -> 'TracksFilter':
        params = self.params.copy()
        params['min_size'] = (params['min_size'].w, params['min_size'].h)
        params['start_time'] = str(params['start_time'])
        params['end_time'] = str(params['end_time'])
        params['event_zone'] = list(params['event_zone'].exterior.coords)
        params['filter_zone'] = list(params['filter_zone'].exterior.coords)
        with open(file_name, 'w') as f:
            json.dump(params, f)
        return self

    def events(self, event_type: str = 'entrance') -> DataFrame:
        """
        Transforms tracks into events
        :param event_type: 'entrance' or 'cash-desk'
        :return: pandas DataFrame with events
        """

        def predicate_in_event(track: dict) -> bool:
            return TrackStats.in_zone(track['trajectory'], self.params['event_zone'], index=0)

        def predicate_out_event(track: dict) -> bool:
            return TrackStats.in_zone(track['trajectory'], self.params['event_zone'], index=-1)

        in_events = self.tracks.filter(predicate_in_event).map(lambda track: {'event_time': track['start'],
                                                                              'track_uuid': track['track_id'],
                                                                              'direction': 'IN'})

        if event_type == 'cash-desk':
            return in_events.to_dataframe(meta={'event_time': str,
                                                'track_uuid': str,
                                                'direction': str}).compute()

        out_events = self.tracks.filter(predicate_out_event).map(lambda track: {'event_time': track['end'],
                                                                                'track_uuid': track['track_id'],
                                                                                'direction': 'OUT'})

        return db.concat([in_events, out_events]).to_dataframe(meta={'event_time': str,
                                                                     'track_uuid': str,
                                                                     'direction': str}).compute()

    def export_events(self, filename: str, events_type: str = 'entrance') -> 'TracksFilter':
        """
        Exports filtered tracks to events
        :param events_type: 'entrance' or 'cash-desk'
        :param filename: file to save events
        """
        self.events(events_type).to_csv(filename, index=False)
        return self

    def filter(self) -> db.Bag:
        """
        Filters tracks by parameters specified in class attributes
        :return: lazy dask object, needs to be computed then
        """

        def predicate_std(track: dict) -> bool:
            return TrackStats.std(track['trajectory'], axis=1) >= self.params['min_y_std']

        def predicate_num_det(track: dict) -> bool:
            return len(track['trajectory']) >= self.params['min_num_det']

        def predicate_zone(track: dict) -> bool:
            if not self.params['use_zone_filter']:
                return True
            return TrackStats.in_zone(track['trajectory'], self.params['filter_zone'],
                                      index=None, min_occurrences=self.params['min_occurrences'])

        def predicate_size(track: dict) -> bool:
            return TrackStats.mean(track['trajectory'], axis=1) > self.params['min_size'].h and \
                   TrackStats.mean(track['trajectory'], axis=0) > self.params['min_size'].w

        def predicate_time(track: dict) -> bool:
            return parse_datetime(track['start']).time() > self.params['start_time'] and \
                   parse_datetime(track['end']).time() < self.params['end_time']

        def criterion(track: dict) -> bool:
            return predicate_time(track) and \
                   predicate_num_det(track) and \
                   predicate_std(track) and \
                   predicate_zone(track) and \
                   predicate_size(track)

        self.tracks = self.tracks.filter(criterion)
        return self

    def compute(self):
        return self.tracks.compute()