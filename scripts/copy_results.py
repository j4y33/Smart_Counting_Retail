from argparse import ArgumentParser
from shutil import copyfile
import pathlib



def get_args():
    parser = ArgumentParser()
    parser.add_argument('--traffic', type=str, required=True, help='Path to traffic')
    parser.add_argument('--processed_videos', type=str, required=True,
                        help='Path to processed_videos')
    parser.add_argument('--tracks', type=str, required=True,
                        help='Path to save tracks')
    parser.add_argument('--dropbox_folder', type=str, required=True, help='Path to dropbox folder')
    parser.add_argument('--reports_folder', type=str, required=True, help='Path to reports_folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    try:
        pathlib.Path(args.dropbox_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.reports_folder).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    copyfile(args.traffic, args.reports_folder + '/traffic.csv')
    copyfile(args.processed_videos, args.reports_folder + '/processed_videos.txt')

    copyfile(args.traffic, args.dropbox_folder + '/traffic.csv')
    copyfile(args.processed_videos, args.dropbox_folder + '/processed_videos.txt')
    copyfile(args.tracks, args.dropbox_folder + '/tracks.json')
