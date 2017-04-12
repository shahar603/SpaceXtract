import cv2
import numpy as np
from math import fabs
import argparse
from os.path import isfile
import simplejson as json
import livestreamer
from collections import OrderedDict
import datetime

KMH_CONVERSION = 3.6
threshold = 0.85
is_small = True

spaceX = [0.185185185, 0.231481481, 0.911458333, 0.963541667, 0.185185185, 0.231481481, 0.807291667, 0.885416667]

#spaceX = [0.185185185, 0.241481481, 0.891458333, 0.963541667, 0.185185185, 0.231481481, 0.807291667, 0.885416667]


def display(frame, velocity_roi, altitude_roi, velocity_array, altitude_array):
    for n, p in velocity_array:
        cv2.rectangle(frame, (int(1920 * spaceX[6] + p), int(1080 * spaceX[4] + 10)),
                      (int(1920 * spaceX[6] + p + 20), int(1080 * spaceX[4] + 45)), (0, 0, 255))
        cv2.putText(frame, str(n), (int(1920 * spaceX[6] + p), int(1080 * spaceX[4])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for n, p in altitude_array:
        cv2.rectangle(frame, (int(1920*spaceX[2]+p), int(1080*spaceX[0]+10)), (int(1920*spaceX[2]+p+20),int(1080*spaceX[0]+45)), (0,0, 255))
        cv2.putText(frame, str(n), (int(1920*spaceX[2]+p), int(1080*spaceX[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    #cv2.imshow('altitude', altitude_roi)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) and False:
        return




def check_result(delta_t,
                 current_velocity, previous_velocity,
                 current_altitude, previous_altitude):
    return delta_t != 0 and \
           fabs((current_velocity-previous_velocity)/delta_t) < 60 and \
           fabs((current_altitude-previous_altitude)/delta_t) < 5 and \
           (previous_velocity == 0 or fabs(current_velocity/previous_velocity) < 4 and fabs(current_velocity/previous_velocity) > 0.3)


def round_to_n_digits(number, n):
    return int(number * 10 ** n) / 10 ** n


def remove_duplicates(array):
    array = list(set(array))

    for i in array:
        for j in array:
            if i != j:
                if fabs(i[1] - j[1]) < 10:
                    array.remove(j)

    return array


def digit_list_to_number(digit_list):
    number = 0

    # Convert the list of digits to a number.
    for digit, _ in digit_list:
        number = 10 * number + digit

    return number


def image_to_digit_list(number_image, templates):
    # Initialize variables.
    digit_list = []
    digit = 0

    # Convert the values from the 'height' and 'velocity' images.
    for digit_image in templates:
        digit_res = cv2.matchTemplate(number_image, digit_image, cv2.TM_CCOEFF_NORMED)

        loc = np.where(digit_res >= threshold)

        for pt in zip(*loc[::-1]):
            digit_list.append((digit, pt[0]))

        digit += 1

    return digit_list


def decimal_point_conversion(current_altitude):
    global is_small

    # Check if the number contains a decimal point.
    if current_altitude >= 900:
        is_small = False

    # Change the value according to the previous check.
    if is_small and current_altitude < 900:
        current_altitude /= 10

    return current_altitude


def calculate_numbers(frame, templates, velocity_roi, altitude_roi):
    # Convert the values from the 'height' and 'velocity' images.
    velocity_array = image_to_digit_list(velocity_roi, templates)
    altitude_array = image_to_digit_list(altitude_roi, templates)

    # Remove duplicate values.
    velocity_array = remove_duplicates(velocity_array)
    altitude_array = remove_duplicates(altitude_array)

    # Make sure the digits were extracted properly.
    if len(velocity_array) != 5 and len(altitude_array) != 3:
        return None, None

    # Sort the digits of the velocity and altitude values by position on the frame.
    velocity_array.sort(key=lambda x: x[1])
    altitude_array.sort(key=lambda x: x[1])

    # Just to show off
    #display(frame, velocity_roi, altitude_roi, velocity_array, altitude_array)

    # Convert the velocity and altitude list to numbers.
    current_velocity = digit_list_to_number(velocity_array)
    current_altitude = digit_list_to_number(altitude_array)

    current_altitude = decimal_point_conversion(current_altitude)

    return current_velocity, current_altitude


def get_frame_size(capture):
    return capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)


def forward_to_launch(cap, time_before_launch):
    number_of_frames = int(cap.get(cv2.CAP_PROP_FPS) * time_before_launch)
    cap.set(cv2.CAP_PROP_POS_FRAMES, number_of_frames)


def crop_frame(frame, frame_height, frame_width, crop_points):
    return frame[int(crop_points[0] * frame_height): int(crop_points[1] * frame_height),
           int(crop_points[2] * frame_width): int(crop_points[3] * frame_width)]


def extract_frames(frame, frame_height, frame_width):
    # Make the frame easier to analyse.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame_thresh = cv2.threshold(frame_gray, 150, 255, cv2.THRESH_BINARY)

    return tuple((crop_frame(frame_thresh, frame_height, frame_width, spaceX[:4]),
                  crop_frame(frame_thresh, frame_height, frame_width, spaceX[4:])))


def output_data(uid, time, velocity, altitude):
    dict_data = OrderedDict([
        ('time', round_to_n_digits(time, 2)),
        ('velocity', round_to_n_digits(velocity, 2)),
        ('altitude', round_to_n_digits(altitude, 2))])

    return json.dumps(
        {
            'uid': uid,
            'timestamp': datetime.datetime.utcnow().isoformat()+'Z',
            'data': dict_data
        }
    )


def analyze_capture(cap, file, templates, launch_time):
    #print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)) - launch_time)

    # Calculate the time each frame takes.
    dt = 1 / cap.get(cv2.CAP_PROP_FPS)

    # Initialize variables.
    current_time = dt
    previous_velocity = 0
    previous_time = 0
    previous_altitude = 0
    data_frq = 1
    start_measurement = False
    frame_index = 0

    # Get frames dimensions.
    frame_width, frame_height = get_frame_size(cap)

    forward_to_launch(cap, launch_time)

    _, frame = cap.read()

    while cap.isOpened():
        #print(round_to_n_digits(frame_index / cap.get(cv2.CAP_PROP_FPS), 2), end='\r')

        _, frame = cap.read()

        if frame is None:
            break

        # Add 1 to frame counter
        frame_index += 1

        # Skip this frame if the rocket hasn't launch yet.
        if start_measurement and frame_index % data_frq != 0:
            current_time += dt
            continue

        # After the rocket launched add dt(time between frames) to the time since launch variable.
        if start_measurement:
            current_time += dt

        # Crop frame
        altitude_roi, velocity_roi = extract_frames(frame, frame_height, frame_width)

        # Get values of velocity and altitude from the current frames.
        current_velocity, current_altitude = calculate_numbers(frame, templates, velocity_roi, altitude_roi)

        # Check if the values make sense.
        if current_velocity is None:
            continue

        # Convert velocity from km/h to m/s.
        current_velocity /= KMH_CONVERSION

        # Modify data to pass validity check if stream doesn't starts from the launch.
        if current_velocity != 0 and current_time == dt:
            previous_velocity = current_velocity
            previous_altitude = current_altitude
            previous_time = current_time - dt

        # Make sure values make sense.
        if (not start_measurement and current_velocity == 0) or not check_result(current_time - previous_time,
                                                                            current_velocity, previous_velocity,
                                                                            current_altitude, previous_altitude):
            continue

        # If this frame is the first frame after launch, start the timer.
        if not start_measurement:
            start_measurement = True

        dict_data = OrderedDict([
            ('time', round_to_n_digits(current_time, 2)),
            ('velocity', round_to_n_digits(current_velocity, 2)),
            ('altitude', round_to_n_digits(current_altitude, 2))])

        #json.dump(dict_data, file)
        file.write(json.dumps(dict_data)+'\n')
        file.flush()

        #print(output_data(frame_index, current_time, current_velocity, current_altitude))
        print(json.dumps(OrderedDict([('uid', frame_index), ('timestamp', datetime.datetime.utcnow().isoformat()+'Z'), ('data', dict_data)])))

        # Update some values.
        previous_altitude = current_altitude
        previous_velocity = current_velocity
        previous_time = current_time

    # Write the last data left in the buffer to the disk.
    #json.dump(dict_data, file)


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Create graphs for SpaceX's launch videos.")
    parser.add_argument('-c', '--capture', action='store', dest='capture_path')
    parser.add_argument('-d', '--destination', action='store', dest='destination_path', default='telemetry.json')
    parser.add_argument('-t', '--templates', action='store', dest='templates_path', default='Images')
    parser.add_argument('-T', '--time', action='store', dest='launch_time', default=0)

    args = parser.parse_args()

    if args.capture_path.startswith('www.youtube.com') or \
            args.capture_path.startswith('http'):
        streams = livestreamer.streams(args.capture_path)
        capture = streams['720p'].url
    else:
        capture = args.capture_path

    # Notify the user if he/she tries to override a file.
    if isfile(args.destination_path):
        if input('{} already exists. Are you sure you want to overwrite it? [y/n]: '.format(
                args.destination_path)) != 'y':
            exit(1)

    # Open output files.
    file = open(args.destination_path, 'wt')

    # Load digits templates from the disk.
    templates = [cv2.imread('{}/{}.jpg'.format(args.templates_path, i), 0) for i in range(10)]

    # Tries to read from video file.
    if args.capture_path is not None:
        cap = cv2.VideoCapture(capture)
    else:
        exit(2)

    # Call analysis function.
    analyze_capture(cap, file, templates, int(args.launch_time))


# Entry point to the program.
if __name__ == '__main__':
    main()
