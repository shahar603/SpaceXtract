import cv2
import extract
import json
from math import fabs
from collections import OrderedDict
import argparse


def check_data(prev_velocity, prev_time, cur_velocity, cur_time):
    return prev_time == cur_time or \
        fabs((cur_velocity-prev_velocity)/(cur_time-prev_time)) < 55



def data_to_json(time, velocity, altitude):
    return json.dumps(OrderedDict(
        [('time', time),
        ('velocity', velocity),
        ('altitude', altitude)]
    ))


def write_to_file(file, string):
    file.write(string + '\n')



def get_data(url, file):
    cap = extract.get_capture(url, '1080p')
    extract.skip_to_launch(cap)
    _, frame = cap.read()

    prev_vel = 0
    prev_time = 0

    time = 0
    dt = 1/cap.get(cv2.CAP_PROP_FPS)

    while frame is not None:
        velocity = extract.calc_velocity(frame)
        altitude = extract.calc_altitude(frame)

        if velocity is not None and altitude is not None and \
                check_data(prev_vel, prev_time, velocity, time):

            write_to_file(file, data_to_json(time, velocity, altitude))
            prev_vel = velocity
            prev_time = time

        _, frame = cap.read()
        time += dt


def set_args():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Create graphs for SpaceX's launch videos.")
    parser.add_argument('-c', '--capture', action='store', dest='capture_path',
                        help='Path (url or local) of the desired video')
    parser.add_argument('-d', '--destination', action='store', dest='destination_path',
                        help='Path to the file that will contain the output')
    parser.add_argument('-T', '--time', action='store', dest='launch_time', default=0,
                        help="Time from launch of the video to the time of the launch (in seconds).\n"
                             "If not given and not live, the capture is set to the launch.\n"
                             "If live, the capture isn't be affected")
    parser.add_argument('-o', action='store_true', dest='out',
                        help='If given results will be printed to stdout')

    args = parser.parse_args()

    if not (args.capture_path or args.destination_path):
        parser.error('No source of destination given, set -c and -d')

    return args


def main():
    args = set_args()
    file = open(args.destination_path, 'w')
    get_data(args.capture_path, file)


if __name__ == '__main__':
    main()
