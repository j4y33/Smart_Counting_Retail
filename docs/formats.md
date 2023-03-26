

### Track Format

```
{
    "camera_id" : uuid    # camera
    "track_id"  : uuid    # track
    "video" : {
        "fps": int,
        "frame_width": int,
        "frame_height": int
    },
    "source"    : str     # result of Path.stem()
    "cluster_id": uuid    # unique person_id from merged tracks into cluster
    "start"     : str     # YYYY-mm-ddTHH:MM:SS.ffff+TZ:00
    "end"       : str     # YYYY-mm-ddTHH:MM:SS.ffff+TZ:00
    "trajectory": [
        {
            "box_id"         : uint64                           # sequential id to match with tracks.data
            "frame_timestamp": str                              # YYYY-mm-ddTHH:MM:SS.ffff+TZ:00
            "bounding_box"   : uint32[]                         # [left, top, width, height]
            "keypoints"      : {                                # Dict[str, Tuple[int, int, float]]
                'right_elbow': [uint32, uint32, float]          # {key: keypoint_name, value: [x, y, score]}
                ...
                'right_knee' : [uint32, uint32, float]
            }
            "confidence"     : float                            # confidence of detection
            "side"           : str                              # one of ['front', 'back']
        }
    ]

Notes:


- track_duration            = end - start                    # seconds
- track_length              = track_duration / video['fps']  # in frames num
- num_detections            = len(trajectory)
- universal_coords_position = x / video['width'], y / video['height']
```

### Event Format
```bash
{
    event_time: "2019-09-06T10:00:00.039910+03:00"
    direction: "IN"
    is_employee: False
    group_id: "976ea5df-d88d-4f5b-8238-e1bda319e4bf"
    smart_counter_uuid: "3bfca449-a207-4c13-9cb5-917a26f3992a"
    track_uuid: "a08a425b-6342-4bec-9315-3393582ba2fd"
    cluster_uuid: "a08a425b-6342-4bec-9315-3393582ba2fd"
}
```

| Field | Type | Description |
| --- | --- | --- |
| event_time | String | Time when person crossed line. Format: [YYYY-mm-ddTHH:MM:SS.ffff+03:00](https://en.wikipedia.org/wiki/ISO_8601) |
| direction | String | Direction of person ```('IN', 'OUT')``` |
| group_id | String | Unique Identifier of group |
| is_employee | Bool | Person is Employee if ```True```, else Visitor |
| smart_counter_uuid | String | Unique Identifier of Region (Database) |
| track_uuid | String | Unique Identifier of person's Track |
| cluster_uuid | String | Unique Identifier of person's Cluster of Tracks |

- traffic.csv
```
{"smart_counter_uuid": 1, "event_time": "2019-02-23T07:48:03.160000+02:00", "direction": "IN", "group_id": "trk-00001", "is_employee": false, "track_uuid": "trk-00001", "cluster_uuid": "1234"}
{"smart_counter_uuid": 1, "event_time": "2019-02-23T07:48:03.160000+02:00", "direction": "IN", "group_id": "trk-00001", "is_employee": false, "track_uuid": "trk-00001", "cluster_uuid": "1234"}
{"smart_counter_uuid": 1, "event_time": "2019-02-23T07:48:03.160000+02:00", "direction": "IN", "group_id": "trk-00001", "is_employee": false, "track_uuid": "trk-00001", "cluster_uuid": "1234"}
{"smart_counter_uuid": 1, "event_time": "2019-02-23T07:48:03.160000+02:00", "direction": "IN", "group_id": "trk-00001", "is_employee": false, "track_uuid": "trk-00001", "cluster_uuid": "1234"}
{"smart_counter_uuid": 1, "event_time": "2019-02-23T07:48:03.160000+02:00", "direction": "IN", "group_id": "trk-00001", "is_employee": false, "track_uuid": "trk-00001", "cluster_uuid": "1234"}
{"smart_counter_uuid": 1, "event_time": "2019-02-23T07:48:03.160000+02:00", "direction": "IN", "group_id": "trk-00001", "is_employee": false, "track_uuid": "trk-00001", "cluster_uuid": "1234"}
{"smart_counter_uuid": 1, "event_time": "2019-02-23T07:48:03.160000+02:00", "direction": "IN", "group_id": "trk-00001", "is_employee": false, "track_uuid": "trk-00001", "cluster_uuid": "1234"}
```
