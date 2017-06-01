import cv2
import extract
import json
from math import fabs
from collections import OrderedDict
import argparse
import os.path
import re


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



def get_data(cap, file, t0, out):
    dt = 1 / cap.get(cv2.CAP_PROP_FPS)
    time = 0
    prev_vel = 0
    prev_time = 0

    extract.skip_from_launch(cap, t0)

    _, frame = cap.read()
    t0, v0, a0 = extract.extract_telemetry(frame)

    if t0 is not None:
        prev_time = t0
        prev_vel = v0
        time = extract.rtnd(t0 + dt, 3)

    while frame is not None:
        velocity = extract.calc_velocity(frame)
        altitude = extract.calc_altitude(frame)

        if velocity is not None and altitude is not None and \
                check_data(prev_vel, prev_time, velocity, time):

            json_data = data_to_json(time, velocity, altitude)

            if out:
                print(data_to_json(time, velocity, altitude))

            write_to_file(file, json_data)
            prev_vel = velocity
            prev_time = time

        _, frame = cap.read()
        time = extract.rtnd(time + dt, 3)



def youtube_url_validation(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    youtube_regex_match = re.match(youtube_regex, url)

    return youtube_regex_match is not None


def create_capture(cap_path):
    if youtube_url_validation(cap_path):
        return extract.get_capture(cap_path, '1080p')

    elif os.path.isfile(cap_path):
        return cv2.VideoCapture(cap_path)

    return None

def to_float(num):
    try:
        return float(num)
    except ValueError:
        return 0


def set_args():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Create graphs for SpaceX's launch videos.")
    parser.add_argument('-c', '--capture', action='store', dest='capture_path',
                        help='Path (url or local) of the desired video')
    parser.add_argument('-d', '--destination', action='store', dest='destination_path',
                        help='Path to the file that will contain the output')
    parser.add_argument('-T', '--time', action='store', dest='launch_time', default=0.0,
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
    cap = create_capture(args.capture_path)

    if cap is None or cap.get(cv2.CAP_PROP_FPS) == 0:
        if youtube_url_validation(args.capture_path):
            print("Cannot access video in URL. Please check the URL is a valid YouTube video")
            exit(2)

        print("Cannot access video in file. Please make sure the path to the file is valid")
        exit(3)

    get_data(cap, file, to_float(args.launch_time), args.out)


if __name__ == '__main__':
    main()
