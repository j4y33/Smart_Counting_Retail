from tools import tracktools
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--tracks', type=str, required=True, help='Path to tracks.json')
    parser.add_argument('--mode', type=str, required=True, choices=['entrance', 'cash-desk'],
                        help='Mode of events')
    parser.add_argument('--output', type=str, required=False, default='traffic.csv',
                        help='Path to save traffic.csv')
    parser.add_argument('--params', type=str, required=False, default={}, help='Path to params.json')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    tracks_filter = tracktools.TracksFilter(args.tracks, params=args.params)
    tracks_filter.compute()
    tracks_filter.export_events(args.output, args.mode)
