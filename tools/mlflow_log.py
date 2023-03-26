from os import makedirs, path  # To create directory with json file
import tools.evaluation as eval
import datetime as dt
import mlflow
import argparse

mlflow.set_tracking_uri("http://10.147.17.161:5000")

"""
Usage:
python mlflow_log.py --ev <path_to_events> --ann <path_to_annotations>
                    --exp <experiment_name> --mode <entrance/cash-desk>
"""


def log_entrance(annotated: str, events: str, delta: int = 10, window: int = 10,
                 experiment: str = 'traffic', params: str = None, run_name: str = '') -> None:
    video = annotated.split('/')[-1].split('.')[0]
    save_path = f'evaluation/{video}/'
    save_file = save_path + 'results.json'

    if not path.exists(save_path):
        makedirs(save_path)

    results = eval.evaluate_counter(events, annotated, mode='entrance', delta=dt.timedelta(minutes=delta),
                                    window=dt.timedelta(seconds=window), save_file=save_file)
    print(results)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('video', video)
        mlflow.log_param('time_split', delta)
        mlflow.log_param('window', window)
        mlflow.log_param('total_in', results['true_in'])
        mlflow.log_param('total_out', results['true_out'])
        mlflow.log_param('average_in', results['true_average_in'])
        mlflow.log_param('average_out', results['true_average_out'])

        mlflow.log_metric('MAE_IN', results['in_mae'])
        mlflow.log_metric('ERROR_IN', results['in_error'])
        mlflow.log_metric('MAE_IN_10', results['in_mae_10'])
        mlflow.log_metric('MAE_OUT', results['out_mae'])
        mlflow.log_metric('ERROR_OUT', results['out_error'])
        mlflow.log_metric('MAE_OUT_10', results['out_mae_10'])
        mlflow.log_metric('PRECISION_IN', results['in_precision'])
        mlflow.log_metric('PRECISION_OUT', results['out_precision'])
        mlflow.log_metric('RECALL_IN', results['in_recall'])
        mlflow.log_metric('RECALL_OUT', results['out_recall'])

        mlflow.log_artifact(annotated)
        mlflow.log_artifact(events)
        mlflow.log_artifact(save_file)
        if params is not None:
            mlflow.log_artifact(params)


def log_cash_desk(annotated: str, events: str, delta: int = 10, window: int = 10,
                  experiment: str = 'cash-desk', params: str = None, run_name: str = '') -> None:
    video = annotated.split('/')[-1].split('.')[0]
    save_path = f'evaluation/{video}/'
    save_file = save_path + 'results.json'

    if not path.exists(save_path):
        makedirs(save_path)

    results = eval.evaluate_counter(events, annotated, mode='cash-desk', delta=dt.timedelta(minutes=delta),
                                    window=dt.timedelta(seconds=window), save_file=save_file)
    print(results)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('video', video)
        mlflow.log_param('time_split', delta)
        mlflow.log_param('window', window)
        mlflow.log_param('total_in', results['true_in'])
        mlflow.log_param('average_in', results['true_average_in'])

        mlflow.log_metric('MAE_IN', results['in_mae'])
        mlflow.log_metric('ERROR_IN', results['in_error'])
        mlflow.log_metric('MAE_IN_10', results['in_mae_10'])
        mlflow.log_metric('PRECISION_IN', results['in_precision'])
        mlflow.log_metric('RECALL_IN', results['in_recall'])

        mlflow.log_artifact(annotated)
        mlflow.log_artifact(events)
        mlflow.log_artifact(save_file)
        if params is not None:
            mlflow.log_artifact(params)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotated', type=str, help='Path to annotated data', required=True)
    parser.add_argument('--events', type=str, help='Path to predicted events', required=True)
    parser.add_argument('--experiment', type=str, help='Name of the experiment', required=True)
    parser.add_argument('--mode', type=str, help='Entrance/cash-desk evaluation', required=True,
                        choices=['cash-desk', 'entrance'])
    parser.add_argument('--delta', type=int, help='Time split in minutes', default=10)
    parser.add_argument('--window', type=int, help='Sliding window value for precision/recall in seconds', default=10)
    parser.add_argument('--params', type=str, help='Path to filter parameters')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'entrance':
        log_entrance(args.annotated, args.events, args.delta, args.window, args.experiment, args.params)
    elif args.mode == 'cash-desk':
        log_cash_desk(args.annotated, args.events, args.delta, args.window, args.experiment, args.params)
    else:
        print('Wrong mode!')
