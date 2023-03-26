import pandas as pd
import datetime as dt
import numpy as np
import json
import argparse
import pytz
from pathlib import Path
from isodate import parse_datetime
from typing import List, Tuple, Union, Callable, Dict, Any

"""
Usage example:
>>>
... delta = dt.timedelta(minutes=10)
... pred = '19/cash-desk/traffic.csv'  # path to events
... ann = '19/cash-desk/2021-1-19-15-58-14_2021-1-19-17-59-33_cash_desk.csv'  # path to annotated data
... print(evaluate_counter(pred, ann, delta=delta))
>>>
"""


def to_common_date(year: int = 2000, month: int = 1, day: int = 1, tz: str = 'Etc/GMT-1') -> Callable:
    def replace_func(date: dt.datetime) -> dt.datetime:
        return date.replace(year=year, month=month, day=day, tzinfo=pytz.timezone(tz))

    return replace_func


def parse_time(file_name: str, *, nformat: str = '%Y-%m-%d-%H-%M-%S') -> Tuple[dt.datetime, dt.datetime]:
    """
    Parses video name into start and end dates (name e.g. 2020-12-23-16-58-23_2020-12-23-19-1-21_entrance.mp4)
    :param file_name: name of a file
    :param nformat: format of the file naming
    :return: start and end dates of a video
    """
    start, end, *_ = file_name.split('/')[-1].split(sep='_')
    start = dt.datetime.strptime(start, nformat)
    end = dt.datetime.strptime(end, nformat)
    return start, end


def frames_to_timedelta(frames: pd.Series, fps: int = 25) -> pd.Series:
    """
    :param frames - series of frame indices
    :param fps - fps rate of the video
    :return - series of timedelta i.e. time passed before the frame (e.g. frame 100 with fps=25 indicates 4sec)
    """
    return (frames / fps).map(lambda f: dt.timedelta(seconds=f) if not np.isnan(f) else f)


def split_time(start: dt.datetime, end: dt.datetime, delta: dt.timedelta) -> List[dt.datetime]:
    """
    Creates bins from :start to :end of duration :delta
    :param start - start date (first boundary)
    :param end - end date (last boundary)
    :param delta - difference between next bin and a previous one
    :return - list of bins
    """
    bins = [start]
    y = bins[-1] + delta
    while y < end:
        bins.append(y)
        y = bins[-1] + delta
    return bins


def calculate_occurrences(event_times: pd.Series, bins: List[dt.datetime]) -> np.ndarray:
    """
    Creates a histogram with number of event occurrences in each bin
    :param event_times - series representing times some event has occurred at
    :param bins - bins that will be compared to event time
    :return - list of number of events occurred in each bin
    """
    hist = []
    for i in range(len(bins) - 1):
        num_occ = len(event_times.loc[(bins[i] < event_times)
                                      & (event_times < bins[i + 1])])
        hist.append(num_occ)
    return np.array(hist)


def get_mae(predictions: np.ndarray, truth: np.ndarray) -> float:
    """
    Calculates MAE value
    :param predictions - predicted values
    :param truth - ground truth values
    :return - MAE score
    """
    return np.absolute((predictions - truth)).mean()


def frames_to_date(start_date: dt.datetime, frames: pd.Series) -> pd.Series:
    """
    Converts frame index to the date it has occurred
    :param start_date - base date
    :param frames - series of frame indices
    :return - series of dates
    """
    return start_date + frames_to_timedelta(frames)


def evaluate_counter(predicted_data: Union[str, pd.DataFrame], annotated_data: Union[str, pd.DataFrame], mode: str,
                     start_time: dt.datetime = None, end_time: dt.datetime = None, save_file=None,
                     delta: dt.timedelta = dt.timedelta(minutes=10), window: dt.timedelta = dt.timedelta(seconds=10)) \
                     -> dict:
    """
    Evaluates people counted in store
    :param predicted_data: pd.DataFrame with data or path to a .csv file
    :param annotated_data: pd.DataFrame with data or path to a .csv file
    :param mode: 'entrance'/'cash-desk'
    :param start_time: explicitly select starting time to evaluate (must be specified with end_time)
    :param end_time: explicitly select ending time to evaluate(must be specified with start_time)
    :param save_file: None if no saving is needed, filename otherwise. If path is specified - folders have to exist
    :param delta: time split for evaluation
    :param window: sliding window of precision/recall algorithm
    :return: dictionary with evaluation resilts
    """
    # Check all the inputs
    if isinstance(predicted_data, str):
        predicted_data = pd.read_csv(predicted_data)
    elif not isinstance(predicted_data, pd.DataFrame):
        raise ValueError('predicted_data must be either a str, or pd.DataFrame type')
    else:
        predicted_data = predicted_data.copy()

    if isinstance(annotated_data, pd.DataFrame):
        if start_time is None or end_time is None:
            raise ValueError('start_time and end_time are not specified and annotated_data is not a filepath')
        annotated_data = annotated_data.copy()
    elif isinstance(annotated_data, str):
        if start_time is None or end_time is None:
            start_time, end_time = parse_time(annotated_data.split('/')[-1])
        annotated_data = pd.read_csv(annotated_data)
    else:
        raise ValueError('annotated_data must be either a str, or pd.DataFrame type')

    if predicted_data.empty:
        return {}

    date_func = to_common_date(2000, 1, 1)
    start_time, end_time = date_func(start_time), date_func(end_time)

    bins = split_time(start_time, end_time, delta)

    results = {}  # Results are stored in a dictionary
    # Convert str date to dt.datetime format and filter irrelevant dates
    predicted_data['event_time'] = predicted_data['event_time'].map(parse_datetime).map(date_func)
    predicted_data = predicted_data.loc[(start_time <= predicted_data['event_time'])
                                        & (predicted_data['event_time'] <= end_time)]

    print(annotated_data)
    true_in_events = frames_to_date(start_time, annotated_data['Номер кадра - Вход'].dropna()).map(date_func)
    predicted_in_events = predicted_data.loc[predicted_data['direction'] == 'IN']['event_time'].map(date_func)

    # Any evaluation type has in events
    true_hist_in = calculate_occurrences(true_in_events, bins)
    predicted_hist_in = calculate_occurrences(predicted_in_events, bins)
    diff_in = predicted_hist_in - true_hist_in
    results['in_mae'] = get_mae(predicted_hist_in, true_hist_in)
    results['in_precision'], results['in_recall'] = \
        precision_recall(predicted_in_events, true_in_events, window)
    results['true_in'] = int(true_in_events.count())
    results['pred_in'] = int(predicted_in_events.count())
    results['true_average_in'] = true_hist_in.mean()
    results['pred_average_in'] = predicted_hist_in.mean()
    results['true_hist_in'] = true_hist_in.tolist()
    results['pred_hist_in'] = predicted_hist_in.tolist()
    results['diff_in'] = diff_in.tolist()
    results['in_mae_10'] = results['in_mae'] * 10 / results['true_average_in']
    results['in_error'] = abs((results['true_in'] - results['pred_in']) / results['true_in'])

    # If out events needed
    if mode == 'entrance':
        predicted_out_events = predicted_data.loc[predicted_data['direction'] == 'OUT']['event_time'].map(date_func)
        true_out_events = frames_to_date(start_time, annotated_data['Номер кадра - Выход'].dropna()).map(date_func)
        true_hist_out = calculate_occurrences(true_out_events, bins)
        predicted_hist_out = calculate_occurrences(predicted_out_events, bins)
        diff_out = predicted_hist_out - true_hist_out
        results['out_mae'] = get_mae(predicted_hist_out, true_hist_out)
        results['out_precision'], results['out_recall'] = \
            precision_recall(predicted_out_events, true_out_events, window)
        results['true_out'] = int(true_out_events.count())
        results['pred_out'] = int(predicted_out_events.count())
        results['true_average_out'] = true_hist_out.mean()
        results['pred_average_out'] = predicted_hist_out.mean()
        results['true_hist_out'] = true_hist_out.tolist()
        results['pred_hist_out'] = predicted_hist_out.tolist()
        results['diff_out'] = diff_out.tolist()

        results['out_mae_10'] = results['out_mae'] * 10 / results['true_average_out']
        results['out_error'] = abs((results['true_out'] - results['pred_out']) / results['true_out'])

    # Save file if needed
    if save_file is not None:
        with open(save_file, 'w') as fp:
            json.dump(results, fp)

    return results


def save(results: Dict[str, Any], filename: str):
    with open(filename, 'w') as f:
        json.dump(results, f)
    print(f"Saved to {filename}")


def precision_recall(predicted: pd.Series, annotated: pd.Series, window: dt.timedelta) -> dict:
    """
    Calculates precision and recall metrics
    :param predicted: series with predicted events (in / out), supposes dates are in dt.datetime format
    :param annotated: series with annotated events respectively, supposes dates are in dt.datetime format
    :param window: window to calculate metrics (e.g. window = 10sec means that
        event is counted as true positive if it was predicted 10sec before/after the actual event)
    :return: precision and recall values respectively
    """
    true_pos = 0
    for event in annotated:
        diff = (predicted - event).abs()
        least_delta = diff.min()
        ind = diff.loc[diff == least_delta].index[0]
        if least_delta < window:
            true_pos += 1
            predicted = predicted.drop(ind)
    false_pos = predicted.count()
    false_neg = len(annotated) - true_pos
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return precision, recall


def _parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate Smart counter accuracy')

    parser.add_argument(
        '--events', required=True,
        help='File with SC video processing results')
    parser.add_argument(
        '--ground_truth', required=True,
        help='File with video file annotations')
    parser.add_argument(
        '--group_time', default=10, type=int,
        help='Grouping time (in minutes) for average stats (default: 10 minutes')
    parser.add_argument(
        '--outfile', default=None,
        help='File path where save result (default: do not save)')
    parser.add_argument(
        '--mode', default=None, choices=['entrance', 'cash-desk'],
        help="'entrance'/'cash-desk'"
    )

    return parser.parse_args()


if __name__ == '__main__':

    # arguments base on previous Evaluation implementation
    # https://github.com/camai-pro/smart-counter/blob/master/scripts/evaluate.py
    args = _parse_arguments()

    group_time = dt.timedelta(minutes=args.group_time)
    if args.mode is None:
        mode = 'cash-desk' if 'cash_desk' in args.ground_truth else 'entrance'
    else:
        mode = args.mode
    results = evaluate_counter(args.events, args.ground_truth, mode, delta=group_time)

    if args.outfile:
        save(results, args.outfile)
    else:
        print(results)
