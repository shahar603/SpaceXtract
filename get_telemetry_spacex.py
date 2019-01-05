import general_extract
import extract_video
import json
from math import fabs
from collections import OrderedDict
import argparse
import os.path
from util import Util, rtnd, to_float

import cv2




DROPPED_FRAME_THRESH = 100




def check_data(prev_velocity, prev_time, cur_velocity, cur_time, prev_alt, cur_alt):
    return prev_time == cur_time or \
           fabs((cur_velocity - prev_velocity) / (cur_time - prev_time)) < 84 and \
           fabs((cur_alt - prev_alt) / (cur_time - prev_time)) < 200 and \
           (fabs(cur_alt - prev_alt) < 8 or cur_time - prev_time > 10)



def data_to_json(time, velocity, altitude):
    return json.dumps(OrderedDict(
        [('time', rtnd(time, 3)),
         ('velocity', rtnd(velocity, 3)),
         ('altitude', altitude)]
    ))


def write_to_file(file, string):
    file.write(string + '\n')









def get_data(cap, file, t0, out, name):
    dt = 1 / cap.get(cv2.CAP_PROP_FPS)

    cur_time = 0
    prev_vel = 0
    prev_altitude = 0
    prev_time = 0
    start = False

    dropped_frames = 0
    con_dropped_frames = 0
    prev_dropped_frame = 0

    time_file = open(name + '.meta', 'w')

    with open('spacex.json', 'r') as spacex_dict_file:
        spacex_dict = json.load(spacex_dict_file)

    session = general_extract.RelativeExtract(spacex_dict, anchor_range=[0, 0.3, 0.75, 1])

    util = Util(session)
    if util.find_anchor(cap, start=0.7, end=1):
        util.skip_from_launch(cap, 'sign', t0)

    _, frame = cap.read()
    _, t0 = session.extract_number(frame, 'time')
    _, v0 = session.extract_number(frame, 'velocity')
    dec, a0 = session.extract_number(frame, 'altitude')

    if dec:
        a0/=10


    if t0 is not None:
        prev_time = t0 - dt
        prev_vel = v0
        prev_altitude = a0
        cur_time = rtnd(t0, 3)

    i = 1

    prev_stage = None



    while frame is not None:
        _, velocity = session.extract_number(frame, 'velocity')
        dec, altitude = session.extract_number(frame, 'altitude')

        if dec and altitude is not None:
            altitude /= 10

        cv2.imshow('frame', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        cur_stage = session.get_template_index(frame, 'stage') + 1

        #if velocity is not None and altitude is not None:
        #    print(cur_time-prev_time, velocity-prev_vel, altitude-prev_altitude)

        if velocity is not None and altitude is not None and \
                (check_data(prev_vel, prev_time, velocity/3.6, cur_time, prev_altitude, altitude)
                 or cur_stage != prev_stage and (cur_stage is not None and cur_time > 60)):

            velocity /= 3.6

            if cur_stage is not None and cur_stage != prev_stage:
                prev_stage = cur_stage

                time_file.write(json.dumps(OrderedDict([
                    ('time', rtnd(cur_time, 3)),
                    ('stage', cur_stage)
                ])) + '\n')
                time_file.flush()

            json_data = data_to_json(cur_time, velocity, altitude)

            if out:
                print(data_to_json(cur_time, velocity, altitude))

            if start:
                write_to_file(file, json_data)

            prev_vel = velocity
            prev_altitude = altitude
            prev_time = cur_time

            if velocity > 0:
                start = True
        else:
            dropped_frames += 1

            if dropped_frames % 10 == 0:
                print('Frame number {} was dropped ({}) which is {:.2f}% of the total frames'.format(i, dropped_frames,
                                                                                                     100 * dropped_frames / i))

        _, frame = cap.read()

        if frame is None:
            break

        if start:
            cur_time += dt

        i += 1

    cv2.destroyAllWindows()
    time_file.close()


def set_args():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Extract telemetry for SpaceX's launch videos.")
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
    parser.add_argument('-f', action='store_true', dest='force',
                        help='Force override of output file')

    args = parser.parse_args()

    if not (args.capture_path or args.destination_path):
        parser.error('No source of destination given, set -c and -d')

    return args


def main():
    args = set_args()

    dest = args.destination_path + '.json'

    if os.path.isfile(dest) and not args.force:
        if input("'%s' already exists. Do you want to override it? [y/n]: " % args.destination_path) != 'y':
            print('exiting')
            exit(4)

    file = open(dest, 'w')
    cap = extract_video.get_capture(args.capture_path)

    if cap is None or cap.get(cv2.CAP_PROP_FPS) == 0:
        if extract_video.youtube_url_validation(args.capture_path):
            print("Cannot access video in URL. Please check the URL is a valid YouTube video")
            exit(2)

        print("Cannot access video in file. Please make sure the path to the file is valid")
        exit(3)

    get_data(cap, file, to_float(args.launch_time), args.out, args.destination_path)


if __name__ == '__main__':
    main()
