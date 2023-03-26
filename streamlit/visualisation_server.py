from __future__ import annotations
import csv
import pytz
import streamlit as st
import yaml
import datetime as dt
import pandas as pd
import os
from pathlib import Path
from dateutil.parser import parse
import re
import base64
from enum import Enum
from typing import Union
from smart_counter.error import ConfigError
import json
import attr


class EventType(Enum):
    IN = 'IN'
    OUT = 'OUT'
    ALL = 'ALL'


class PeriodMode(Enum):
    Day = 'D'
    Week = 'W'
    Month = 'M'
    Year = 'Y'


class TimeCoverage:

    def __init__(self, coverage, start_time, end_time):
        self._coverage = coverage
        self._start_time = start_time
        self._end_time = end_time

    @property
    def coverage(self):
        return self._coverage

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time


class DataItem():

    def __init__(self, date: dt.date, processed_videos_file: Union[Path, str], traffic_file: Path, tz: pytz.tzfile.tz):
        """
        Parameters
        ----------
        date
        processed_videos_file
            file with all processed videos pathes
        traffic_file : Path
            file with all traffic events
        """
        self._date = date
        self._processed_videos_file = processed_videos_file
        self._traffic_file = traffic_file
        self._tz = tz

    @property
    def tz(self) -> pytz.tzfile.tz:
        return self._tz

    @property
    def date(self) -> dt.date:
        return self._date

    @property
    def events(self) -> pd.DataFrame:
        """
        get all events

        Returns
        -------
            all events as pd.DataFrame
        """
        return pd.read_csv(self._traffic_file, converters={'event_time': parse})

    @property
    def videos(self) -> [str]:
        """
        get all processed videos names

        Returns
        -------
            list of names
        """
        return open(self._processed_videos_file, 'r').readlines()


class Report():
    HOURS_IN_DAY = 24

    def __init__(self, items: [DataItem], period_start_date: dt.date = None, period_end_date: dt.date = None,
                 start_working_time: dt.time = None,
                 end_working_time: dt.time = None,
                 events: Union[pd.DataFrame, pd.Series] = None):
        self._items = items
        self._period_start_date = period_start_date
        self._period_end_date = period_end_date
        self._start_working_time = start_working_time
        self._end_working_time = end_working_time

        self._events = events
        self._time_coverage = None

    def select(self, params: dict) -> Report:
        """
        select events from data items by params

        Parameters
        ----------
        params
            date_from : dt.datetime
            date_to : dt.datetime
            start_working_time : dt.time
            end_working_time : dt.time
            event_type (traffic): EventType
        Returns
        -------
            self
        """
        if params.get('date_from', '') == '' and params.get('date_to', '') == '':
            raise ConfigError(f'one of date_from or date_to shoul be passed as param to select')

        period_start_date = params.get('date_from', dt.date(2000, 1, 1))
        period_end_date = params.get('date_to', dt.date(2030, 1, 1))

        data_items_to_process = [d for d in self._items if period_end_date >= d.date >= period_start_date]

        if len(data_items_to_process) == 0:
            return Report(None)

        start_working_time = params.get('start_working_time', dt.time(8, 0, 0))
        end_working_time = params.get('end_working_time', dt.time(22, 0, 0))

        events = [f.events for f in data_items_to_process]
        events = [ev[(end_working_time >= ev['event_time'].dt.time) & (
            ev['event_time'].dt.time >= start_working_time)] for ev in events]
        events = pd.concat(events, ignore_index=True)

        events = events[events.direction == params['event_type'].value
                        ] if params.get('event_type', EventType.ALL) != EventType.ALL else events

        return Report(data_items_to_process,
                      period_start_date,
                      period_end_date,
                      start_working_time,
                      end_working_time,
                      events)

    def group_by(self, params: dict) -> Report:
        """
        group events

        Parameters
        ----------
        params
            time: PeriodMode

        Returns
        -------
            self
        """
        if self._events is None:
            return Report(None)

        mode = params.get('time', PeriodMode.Month)
        events_dt = self._events.event_time.dt if mode == PeriodMode.Day else pd.to_datetime(
            self._events.event_time, utc=True).dt
        grouping_param = (
            events_dt.hour if mode == PeriodMode.Day
            else events_dt.date if mode == PeriodMode.Week
            else events_dt.day
        )

        first_event = self._events.event_time.iloc[0]
        grouping_range = (
            [d for d in range(self.HOURS_IN_DAY)] if mode == PeriodMode.Day
            else [
                d.strftime('%Y-%m-%d') for d in pd.date_range(
                    first_event.to_period(mode.value).start_time,
                    first_event.to_period(mode.value).end_time
                )
            ] if mode == PeriodMode.Week
            else [d+1 for d in range(first_event.days_in_month)]
        )

        events = (self._events['event_time'].groupby(grouping_param).count())
        events.index = events.index.map(lambda x: x.strftime(
            '%Y-%m-%d')) if mode == PeriodMode.Week else events.index

        empty_range = pd.Series([0]*len(grouping_range), index=grouping_range)
        events = events.combine(empty_range, max, fill_value=0)
        return Report(self._items,
                      self._period_start_date,
                      self._period_end_date,
                      self._start_working_time,
                      self._end_working_time, events)

    @property
    def videos_count(self) -> int:
        if self._time_coverage is None:
            self._time_coverage = self._count_time_coverage()
        return int(self._time_coverage.coverage.sum())

    @property
    def events_count(self) -> int:
        if self._events is None:
            return 0
        return int(self._events.sum())

    @property
    def traffic(self) -> pd.DataFrame:
        return self._events

    @property
    def time_coverage(self) -> pd.DataFrame:
        """
        get time coverage

        Returns
        -------
            time coverage
        """
        if self._time_coverage is None:
            self._time_coverage = self._count_time_coverage()
        return self._time_coverage

    def _count_time_coverage(self):
        if len(self._items) == 0:
            return None
        all_stamps = sorted(self._get_all_videos_timestamps(self._items))
        coverage = {t: 1 for t in all_stamps if self._end_working_time >= t.time() >= self._start_working_time}
        coverage[pytz.utc.localize(self._period_start_date)] = 0
        coverage[pytz.utc.localize(self._period_end_date)] = 0

        coverage = pd.DataFrame.from_dict(coverage, orient='index')
        start_time = all_stamps[0]
        end_time = all_stamps[-1]

        return TimeCoverage(coverage,
                            start_time,
                            end_time)

    @staticmethod
    def _get_datetime_from_video_name(video: str, tz: pytz.tzfile.tz) -> dt.datetime:
        """
        get datetime from video name

        Parameters
        ----------
        video
            video name
        tz
            timezone

        Returns
        -------
            datetime of video
        """
        stamp = dt.datetime.fromtimestamp(int(video.split('_')[0]), tz=tz)
        return stamp.replace(tzinfo=pytz.utc)

    @staticmethod
    def _get_all_videos_timestamps(dates: [DataItem]) -> [dt.datetime]:
        """
        get all datetimes from processed videos

        Parameters
        ----------
        dates
            data items
        tz
            timezone

        Returns
        -------
            list of the same size as dates with datetimes
        """
        return [Report._get_datetime_from_video_name(video, f.tz)
                for f in dates
                for video in f.videos]


def download_link(df: pd.DataFrame, start_date: dt.date, end_date: dt.date) -> str:
    """
    Generates a link to download traffic events.
    """
    traffic = df.to_csv(index=False)
    b64 = base64.b64encode(traffic.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" \
        download="traffic_{start_date}_{end_date}.csv">Export as CSV</a>'


def get_all_dates(store_folder: Union[Path, str]) -> pd.DatetimeIndex:
    """
    get all existing dates from folder names

    Returns
    -------
        pandas index with all dates
    """
    return pd.DatetimeIndex(
        dt.datetime.strptime(i, '%Y%m%d').date() for i in sorted(os.listdir(store_folder)) if re.match('\d{8}', i)
    )


def get_available_periods(mode: PeriodMode, store_folder: Union[Path, str], period: pd.Period = None) -> pd.Period:
    """
    get possible smaller periods in bigger one

    Parameters
    ----------
    mode
        PeriodMode
    period
        bigger period, None for year

    Returns
    -------
        all possible periods in this period
    """
    res = sorted(set(get_all_dates(store_folder).to_period(mode.value)))
    res = [p for p in res if period_condition(mode, p, period)]
    res.reverse()
    return res


def period_condition(mode: PeriodMode, period_to_check: pd.Period, period: pd.Period = None) -> bool:
    """
    check if period is inside another period

    Parameters
    ----------
    mode
        PeriodMode
    period_to_check
    period
        period to check with, None for year

    Returns
    -------
        True if period_to_check is inside period, otherwise False
    """
    if mode == PeriodMode.Year:
        return True
    if period is None:
        return True
    if mode == PeriodMode.Month:
        return period_to_check.year == period.year
    if mode in [PeriodMode.Week, PeriodMode.Day]:
        return period_to_check.start_time.year == period.year and period_to_check.start_time.month == period.month


def show_cross_graphic_of_reports(report_entrance: Report,
                                  report_cash_desk: Report,
                                  labels: [str] = ["entrance", "cash-desk"]):
    """
    show two reports on one graphic

    Parameters
    ----------
    report_entrance
    report_cash_desk
    """
    st.header("Visitor vs. Buyer Traffic")
    if report_entrance.traffic is None or report_cash_desk.traffic is None:
        "No some data"
        return

    joint_traffic = pd.concat([report_entrance.traffic, report_cash_desk.traffic],
                              axis=1, keys=labels)

    st.line_chart(joint_traffic)


def show_date(report: Report, mode: PeriodMode):
    start_datetime = report.time_coverage.start_time
    end_datetime = report.time_coverage.end_time
    if mode == PeriodMode.Day:
        'Date: ', start_datetime.date()
        col1, col2 = st.beta_columns(2)
        with col1:
            'Time from: ', start_datetime.time()
        with col2:
            'Time to: ', end_datetime.time()
        return

    col1, col2 = st.beta_columns(2)
    with col1:
        'Date from: ', start_datetime.date()
    with col2:
        'Date to: ', end_datetime.date()
    return


def show_report(report: Report, mode: PeriodMode, title: str):
    """
    show report graphics

    Parameters
    ----------
    report
    title
        title of section
    """
    st.header(title)
    if report.traffic is None:
        'No data'
    else:

        'Number of videos processed: ', report.videos_count
        show_date(report, mode)
        ''
        'Time Coverage'
        st.bar_chart(report.time_coverage.coverage)
        'Traffic: ', report.events_count
        st.bar_chart(report.traffic)

        st.markdown(download_link(report.traffic,
                                  report.time_coverage.start_time.date(),
                                  report.time_coverage.end_time.date()), unsafe_allow_html=True)


def get_report(store_folder: Union[Path, str],
               period: pd.Period,
               camera: str = 'entrance',
               mode: PeriodMode = PeriodMode.Day,
               start_working_time: dt.time = dt.time(8, 0, 0),
               end_working_time: dt.time = dt.time(22, 0, 0)) -> Report:
    """
    get report for certain camera and period

    Parameters
    ----------
    store_folder
        folder with all data
    camera
    period
    mode
        PeriodMode
    tz
        timezone

    Returns
    -------
        report
    """
    items = get_all_data_items(store_folder, camera)

    return Report(items).select(params={
        'date_from': period.start_time,
        'date_to': period.end_time,
        'start_working_time': start_working_time,
        'end_working_time': end_working_time,
        'event_type': EventType.IN
    }).group_by(params={'time': mode})


def period_select_box(title: str,
                      mode: PeriodMode,
                      store_folder: Union[Path, str],
                      period: pd.Period = None) -> pd.Period:
    return st.selectbox(title, get_available_periods(mode, store_folder, period))


def get_all_data_items(store_folder: Union[Path, str], camera: str) -> [DataItem]:
    """
    get all valid DataItems from folder

    Parameters
    ----------
    store_folder
        Path to folder with all reports
    camera
        camera name

    Returns
    -------
        list of valid DataItem
    """
    res = []
    for folder in os.listdir(store_folder):
        path_to_data = store_folder / folder / camera
        processed_videos = path_to_data / 'processed_videos.txt'
        traffic = path_to_data / 'traffic.csv'
        if (not os.path.exists(processed_videos)
            or not os.path.exists(traffic)
                or not re.match('\d{8}', folder)):
            continue

        tz = pd.read_csv(traffic, nrows=1, converters={'event_time': lambda x: parse(x).tzinfo}).event_time
        if tz.empty:
            continue
        res.append(DataItem(
            dt.datetime.strptime(folder, '%Y%m%d'),
            processed_videos,
            traffic,
            tz[0]))
    return res


def main():
    # get config params
    params = yaml.safe_load(open('streamlit/params.yaml', 'r'))

    # title
    st.title('Miniso Reports')

    # starting params section
    st.header('Params')

    # folders with data for each store
    stores_folder = params.get('pathes_to_stores', {})

    # choose store
    store = st.selectbox('choose store:', list(stores_folder.keys()))
    store_folder = Path(stores_folder[store])

    col_start_working_time, col_end_working_time = st.beta_columns(2)
    with col_start_working_time:
        start_working_time = st.time_input('start working time', dt.time(8, 0, 0))
    with col_end_working_time:
        end_working_time = st.time_input('end working time', dt.time(22, 0, 0))
    # choose mode
    mode = st.radio("choose mode", (PeriodMode.Day, PeriodMode.Week, PeriodMode.Month),
                    format_func=lambda x: PeriodMode(x).name)

    # choose dates
    column_year, column_month, column_week_day = st.beta_columns(3)
    with column_year:
        period = period_select_box("choose year", PeriodMode.Year, store_folder)

    if mode in [PeriodMode.Month, PeriodMode.Week, store_folder, PeriodMode.Day]:
        with column_month:
            period = period_select_box("choose month", PeriodMode.Month, store_folder, period)
        with column_week_day:
            if mode == PeriodMode.Week:
                period = period_select_box("choose week", PeriodMode.Week, store_folder, period)
            if mode == PeriodMode.Day:
                period = period_select_box("choose day", PeriodMode.Day, store_folder, period)

    # show reports
    report_entrance = get_report(store_folder, period, 'entrance', mode, start_working_time, end_working_time)
    show_report(report_entrance, mode, 'Visitor traffic')

    if mode == PeriodMode.Day and period.start_time.dayofweek == 6:
        st.header("Buyers traffic")
        'No data is expected for Sunday for cash-desk camera'
        st.header("Visitor vs. Buyer Traffic")
        'No some data'
        return

    report_cash_desk = get_report(store_folder,  period, 'cash-desk', mode, start_working_time, end_working_time)
    show_report(report_cash_desk, mode, 'Buyers traffic')

    show_cross_graphic_of_reports(report_entrance, report_cash_desk)


try:
    main()
except Exception as err:
    'Error: ', err
