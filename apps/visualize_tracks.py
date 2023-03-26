import datetime as dt
import os
import re
import json
import tempfile
import base64

import pandas as pd
import streamlit as st
from isodate import parse_time
from PIL import Image
from dask.distributed import Client

import SessionState as ss

import tools.mlflow_log as mlflow_export
from tools import evaluation as eval
from tools.tracktools import TracksDrawer, TracksFilter

params = {"min_num_det": 1,
          "min_y_std": 0,
          "min_size": [0, 0],
          "event_zone": [[700.0, 0.0], [1270.0, 0.0], [1270.0, 370.0], [700.0, 370.0], [700.0, 0.0]],
          "filter_zone": [[700.0, 0.0], [1270.0, 0.0], [1270.0, 370.0], [700.0, 370.0], [700.0, 0.0]],
          "use_zone_filter": False,
          "start_time": "06:00:00",
          "end_time": "23:45:00"}


@st.cache
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.
    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    if isinstance(object_to_download, dict):
        object_to_download = json.dumps(object_to_download)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


if __name__ == '__main__':

    session_state = ss.get(tracks_file=None, ann_file=None, tracks=None, picture=None, event_type='entrance',
                           events=None, tracks_filter=None, params=None, client=None, iter_step=1)

    if session_state.client is None:
        session_state.client = Client(processes=False)

    tracks_file = st.text_input('Path to tracks.json', value='/tracks.json')

    picture_file = st.file_uploader('Upload background picture', type=['.png', '.jpg', 'jpeg'])
    params_file = st.sidebar.file_uploader('Load parameters.json', type=['.json'])
    if params_file is not None:
        params = json.load(params_file)
    st.sidebar.header('Draw each N detections')
    step = int(st.sidebar.text_input('Iteration step', value=1))

    st.sidebar.header('By num detections')
    min_num_det = int(st.sidebar.text_input('Minimum number of detections in track', value=params['min_num_det']))

    st.sidebar.header('By Y-axis standard deviation')
    min_y_std = int(st.sidebar.text_input('Minimum std', value=params['min_y_std']))

    st.sidebar.header('By median box size')
    size_str = st.sidebar.text_input('Minimum box size (w, h)', value=', '.join(map(str, params['min_size'])))
    min_size = tuple(map(int, size_str.replace(' ', '').split(',')))

    st.sidebar.header('Zone')
    zone_regex = r'\d+\s\d+'
    zone_str_to_list = lambda s: tuple(map(int, s.split()))
    poly_to_str = lambda zone: ', '.join([' '.join(map(str, map(int, tup))) for tup in zone])
    event_zone_str = st.sidebar.text_input('Event zone (polygon)', value=poly_to_str(params['event_zone']))
    event_zone = list(map(zone_str_to_list, re.findall(zone_regex, event_zone_str)))
    filter_zone_str = st.sidebar.text_input('Filter zone (polygon)', value=poly_to_str(params['filter_zone']))
    filter_zone = list(map(zone_str_to_list, re.findall(zone_regex, filter_zone_str)))
    filter_zone_on = st.sidebar.checkbox('Use zone filter', value=False)
    min_occurrences = int(st.sidebar.text_input('Min detections in zone', value=0))

    st.sidebar.header('By time')
    min_time, max_time = st.sidebar.slider('Time range', min_value=dt.datetime(2020, 1, 1, 6, 0, 0).time(),
                                           max_value=dt.datetime(2020, 1, 1, 23, 45, 0).time(),
                                           value=[parse_time(params['start_time']),
                                                  parse_time(params['end_time'])],
                                           step=dt.timedelta(minutes=15),
                                           format='HH:mm')

    st.sidebar.header('Drawing options')
    show_arrows = st.sidebar.checkbox('Draw arrows', value=True)
    location = 'head'
    if show_arrows:
        location = st.sidebar.selectbox('Keypoint location', ['head', 'center'])
    show_boxes = st.sidebar.checkbox('Draw bounding boxes', value=False)
    show_filter_zone = st.sidebar.checkbox('Draw filter zone', value=False)
    show_event_zone = st.sidebar.checkbox('Draw event zone', value=False)

    image_display = st.empty()
    picture = None
    picture_size = None
    # To draw picture even if there are no tracks
    if picture_file is not None:
        picture = Image.open(picture_file)
        picture_size = picture.size
        image_display.image(picture, use_column_width=True)

    if not os.path.exists(tracks_file):
        st.text('No file found')
    else:
        params = {'min_num_det': min_num_det,
                  'min_y_std': min_y_std,
                  'min_occurrences': min_occurrences,
                  'min_size': min_size,
                  'event_zone': event_zone,
                  'filter_zone': filter_zone,
                  'use_zone_filter': filter_zone_on,
                  'start_time': str(min_time),
                  'end_time': str(max_time)}
        event_type = st.selectbox('Select events type', options=['entrance', 'cash-desk'])

        # If any filters have changed or tracks were not yet initialized
        if session_state.tracks_filter is None or session_state.params != params \
            or tracks_file != session_state.tracks_file:
            session_state.tracks_file = tracks_file
            session_state.params = params
            session_state.tracks_filter = TracksFilter(tracks_file, params=params, picture_size=picture_size)
            session_state.tracks = session_state.tracks_filter.filter().compute()
            session_state.events = session_state.tracks_filter.events(event_type)
            session_state.picture = None  # To indicate that we'll have to redraw the image

        # To update events if event_type has changed
        if session_state.event_type != event_type:
            session_state.event_type = event_type
            session_state.events = session_state.tracks_filter.events(event_type)

        st.text(f'Number of tracks: {len(session_state.tracks)}')
        st.text(f"Num events : {len(session_state.events)}")

        # If picture is uploaded
        # At this points there are tracks for sure, so we just have to draw them
        if picture_file is not None:
            # If filters have changed
            if session_state.picture is None or step != session_state.iter_step:
                session_state.iter_step = step
                session_state.picture = TracksDrawer(picture).draw(session_state.tracks, step=step,
                                                                   show_arrows=show_arrows,
                                                                   show_boxes=show_boxes,
                                                                   filter_zone=filter_zone if show_filter_zone else None,
                                                                   event_zone=event_zone if show_event_zone else None,
                                                                   location=location)
            image_display.image(session_state.picture, use_column_width=True)

        # Just some links for downloads
        tmp_download_link = download_link(params, 'params.json', 'Export params as JSON')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

        tmp_download_link = download_link(session_state.events, 'events.csv', 'Export events as CSV')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

        annotated_data = st.file_uploader('Select file with annotations',
                                          type=['.csv'], accept_multiple_files=False)

        if annotated_data is not None:
            annotated = pd.read_csv(annotated_data)

            split = int(st.text_input('Enter time split in minutes', value=10))
            delta = dt.timedelta(minutes=split)
            start_time, end_time = eval.parse_time(annotated_data.name)
            results = eval.evaluate_counter(session_state.events, annotated, mode=event_type, start_time=start_time,
                                            end_time=end_time, delta=delta)
            if st.checkbox('Export to MLFlow'):
                exp_name = st.text_input('Experiment name', value='visualization-experiment')
                run_name = st.text_input('Run name', value='')
                if st.button('Export'):
                    with tempfile.NamedTemporaryFile(prefix=annotated_data.name, suffix='.csv', delete=True) as t1, \
                        tempfile.NamedTemporaryFile(prefix='events', suffix='.csv', delete=True) as t2, \
                        tempfile.NamedTemporaryFile(prefix='params', suffix='.json', delete=True) as t3:

                        annotated.to_csv(t1.name)
                        session_state.events.to_csv(t2.name)
                        session_state.tracks_filter.export_params(t3.name)
                        if event_type == 'entrance':
                            mlflow_export.log_entrance(t1.name, t2.name, params=t3.name, delta=split,
                                                       run_name=run_name, experiment=exp_name)
                        elif event_type == 'cash-desk':
                            mlflow_export.log_cash_desk(t1.name, t2.name, params=t3.name, delta=split,
                                                        run_name=run_name, experiment=exp_name)

            st.json(results)
