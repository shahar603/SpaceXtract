import sys

from SpaceXtract import general_extract
from SpaceXtract import extract_video
from SpaceXtract.util import Util, rtnd, to_float
import json
from math import fabs
from collections import OrderedDict
import argparse
import os.path

import cv2



CONFIG_FILE_PATH = '../ConfigFiles/rocketlab/rocketlab_20plus.json'


KMH = 3.6
DECIMAL_CONVERSION = 10
DROPPED_FRAME_THRESH = 100
PRECISION = 3
ANCHOR_SEARCH_START_TIME_FRACTION = 0.7
ANCHOR_SEARCH_END_TIME_FRACTION = 1
LAUNCH_VELOCITY = 0

SCALE = 0.5


def check_data(prev_velocity, prev_time, cur_velocity, cur_time, prev_alt, cur_alt):
    return prev_time == cur_time or \
           fabs((cur_velocity - prev_velocity) / (cur_time - prev_time)) < 84 and \
           fabs((cur_alt - prev_alt) / (cur_time - prev_time)) < 200 and \
           (fabs(cur_alt - prev_alt) < 8 or cur_time - prev_time > 10)


def check_stage_switch(cur_stage, prev_stage):
    return cur_stage != prev_stage and (cur_stage is not None and cur_time > 60)
           
           
           

def data_to_json(time, velocity, altitude):
    return json.dumps(OrderedDict(
        [('time', rtnd(time, PRECISION)),
         ('velocity', rtnd(velocity, PRECISION)),
         ('altitude', altitude)]
    ))


def write_to_file(file, string):
    file.write(string + '\n')


def show_frame(frame, session, rocketlab_dict_frame : dict):
    global counter
    location= session.anchor_location
    x_rel = int(location[2]* 1920)
    y_rel = int(location[0]*1080)
    p1_anchor = (x_rel,y_rel)
    p2_anchor = (int(location[3]* 1920),int(location[1]*1080))

    time_location = session.ratio_to_pixel(session.relative_to_abs(rocketlab_dict_frame["sign"][0]),frame.shape)
    p1_time = (int(time_location[2]),int(time_location[0]))
    p2_time = (int(time_location[3]),int(time_location[1]))

    alt_location = session.ratio_to_pixel(session.relative_to_abs(rocketlab_dict_frame["altitude"][0]),frame.shape)
    p1_alt = (int(alt_location[2]), int(alt_location[0]))
    p2_alt = (int(alt_location[3]), int(alt_location[1]))

    speed_location = session.ratio_to_pixel(session.relative_to_abs(rocketlab_dict_frame["velocity"][0]),frame.shape)
    p2_speed = (int(speed_location[2]), int(speed_location[0]))
    p1_speed = (int(speed_location[3]), int(speed_location[1]))

    t_location = session.ratio_to_pixel(session.relative_to_abs(rocketlab_dict_frame["time"][0]), frame.shape)
    t2_speed = (int(t_location[2]), int(t_location[0]))
    t1_speed = (int(t_location[3]), int(t_location[1]))

    cv2.rectangle(frame,p1_time,p2_time,(255,0,0), 2)
    cv2.rectangle(frame,p1_anchor,p2_anchor,(255,0,0), 2)
    cv2.rectangle(frame,p1_alt,p2_alt,(255,0,0), 2)
    cv2.rectangle(frame,p1_speed,p2_speed,(255,0,0), 2)
    cv2.rectangle(frame,t1_speed,t2_speed,(255,0,0), 2)
    cv2.imshow('frame', cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE))




def decimal_point_conversion(digit_pos_list):
    """
    :param digit_pos_list: a list of the digits with their position.
    :return:
    """
    return True


def get_data(cap, file, t0, out, name, live, from_launch=True):
    dt = 1 / cap.get(cv2.CAP_PROP_FPS)

    cur_time = 0
    prev_vel = 0
    prev_altitude = 0
    prev_time = 0
    start = False
    dropped_frames = 0
    frame_index = 1
    prev_stage = None
    
    

    time_file = open(name + '.meta', 'w')

    with open(CONFIG_FILE_PATH, 'r') as spacex_dict_file:
        s= spacex_dict_file.read()
        rocketlab_dict = json.loads(s)
        unchanged_dict= json.loads(s)


    session = general_extract.RelativeExtract(rocketlab_dict, anchor_range=rocketlab_dict['anchor'][0])
    util = Util(session)
    
    
    if not live:
        if from_launch and t0 is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t0*cap.get(cv2.CAP_PROP_FPS))
        elif util.find_anchor(cap, start=ANCHOR_SEARCH_START_TIME_FRACTION, end=ANCHOR_SEARCH_END_TIME_FRACTION):
            util.skip_from_launch(cap, 'sign', t0)
    else:
        util.play_until_anchor_found(cap, session)
    
    _, frame = cap.read()
        
    _, t0 = session.extract_number(frame, 'time', decimal_point_conversion)
    _, v0 = session.extract_number(frame, 'velocity', decimal_point_conversion)
    dec, a0 = session.extract_number(frame, 'altitude', decimal_point_conversion)    

    if dec:
        a0 /= DECIMAL_CONVERSION

    if t0 is not None:
        prev_time = t0 - dt
        prev_vel = v0 / KMH
        prev_altitude = a0
        cur_time = rtnd(t0, 3)

    
    while frame is not None:
        _, velocity = session.extract_number(frame, 'velocity', decimal_point_conversion)
        dec, altitude = session.extract_number(frame, 'altitude', decimal_point_conversion)

        if dec and altitude is not None:
            altitude /= DECIMAL_CONVERSION

        show_frame(frame,session,unchanged_dict)
            
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        cur_stage = 2

        if velocity is not None and altitude is not None and \
                (check_data(prev_vel, prev_time, velocity/KMH, cur_time, prev_altitude, altitude)
                 or check_stage_switch(cur_stage, prev_stage)):

            velocity /= KMH

            if cur_stage is not None and cur_stage != prev_stage:
                prev_stage = cur_stage

                time_file.write(json.dumps(OrderedDict([
                    ('time', rtnd(cur_time, PRECISION)),
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

            if velocity > LAUNCH_VELOCITY:
                start = True
        else:
            dropped_frames += 1

            if dropped_frames % DROPPED_FRAME_THRESH == 0:
                print('Frame number {} was dropped ({}) which is {:.2f}% of the total frames'.format(frame_index,
                    dropped_frames,
                    100 * dropped_frames / frame_index))

        _, frame = cap.read()

        if start:
            cur_time += dt

        frame_index += 1

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
    parser.add_argument('-l', action='store_true', dest='live',
                        help='Is the source live')
    parser.add_argument('-s', action='store_true', dest='from_launch',
                        help='Start from time')

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

    get_data(cap, file, to_float(args.launch_time), args.out, args.destination_path, args.live, args.from_launch)


if __name__ == '__main__':
    main()
