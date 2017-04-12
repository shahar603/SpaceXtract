import cv2
import numpy as np
from math import fabs
import argparse
from os.path import isfile
import simplejson as json
import livestreamer
from collections import OrderedDict
import datetime

KMH_CONVERSION = 3.6 # m/s
threshold = 0.85 # Minimum probability for the OCR algorithm. 0.85 for 1080p, not live capture.
is_small = True # Does the altitude values contains a decimal point.


# A list of ratios that contains the position of the altitude and velocity in the frame.
# Altitude:   [top]     [bottom]       [left]       [right]
# Velocity:   [top]     [bottom]       [left]       [right]
spaceX = [0.185185185, 0.231481481, 0.911458333, 0.963541667,
          0.185185185, 0.231481481, 0.807291667, 0.885416667]

#spaceX = [0.185185185, 0.241481481, 0.891458333, 0.963541667, 0.185185185, 0.231481481, 0.807291667, 0.885416667]


# Show the frames with the digits marked only used for debugging.
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





# Returns True if the velocity and altitude make sense.
def check_result(delta_t, cur_velocity, prev_velocity, cur_altitude, prev_altitude):
    return delta_t != 0 and \
           fabs((cur_velocity - prev_velocity) / delta_t) < 60 and \
           fabs((cur_altitude - prev_altitude) / delta_t) < 5 and \
           (prev_velocity == 0 or fabs(cur_velocity / prev_velocity) < 4 and fabs(cur_velocity / prev_velocity) > 0.3)




# Returns the input number with n digit after the decimal point.
def round_to_n_digits(number, n):
    return int(number * 10 ** n) / 10 ** n




# Remove digits found twice or more by the OCR algorithm.
def remove_duplicates(array):
    # Removes duplicate elements.
    array = list(set(array))

    for i in array:
        for j in array:
            if i != j:
                # Remove values too close together (less than 10 pixels apart).
                if fabs(i[1] - j[1]) < 10:
                    array.remove(j)

    return array



# Convert a list of digits to a number.
def digit_list_to_number(digit_list):
    number = 0

    # Convert the list of digits to a number.
    for digit, _ in digit_list:
        number = 10 * number + digit

    return number




# Convert an image to a list of digits.
def image_to_digit_list(number_image, templates):
    # Initialize variables.
    digit_list = []
    digit = 0

    # Convert the values from the 'height' and 'velocity' images.
    for digit_image in templates:
        # Get a matrix of values containing the probability the pixel is the top-left part of the template.
        digit_res = cv2.matchTemplate(number_image, digit_image, cv2.TM_CCOEFF_NORMED)

        # Get a list of all the pixels that have a probability >= to the threshold.
        loc = np.where(digit_res >= threshold)

        # Create a list that contains the x position and the digit.
        for pt in zip(*loc[::-1]):
            digit_list.append((digit, pt[0]))

        digit += 1

    return digit_list





# Change the altitude value acourding to the position of the decimal point.
def decimal_point_conversion(altitude):
    global is_small

    # Check if the number contains a decimal point.
    if altitude >= 900:
        is_small = False

    # Change the value according to the previous check.
    if is_small and altitude < 900:
        altitude /= 10

    return altitude



# Get images of the velocity and altitude and return their numeric values.
def calculate_numbers(frame, templates, velocity_roi, altitude_roi):
    # Convert the values from the 'height' and 'velocity' images.
    velocity_list = image_to_digit_list(velocity_roi, templates)
    altitude_list = image_to_digit_list(altitude_roi, templates)

    # Remove duplicate values.
    velocity_list = remove_duplicates(velocity_list)
    altitude_list = remove_duplicates(altitude_list)

    # Make sure the digits were extracted properly.
    if len(velocity_list) != 5 and len(altitude_list) != 3:
        return None, None

    # Sort the digits of the velocity and altitude values by position on the frame.
    velocity_list.sort(key=lambda x: x[1])
    altitude_list.sort(key=lambda x: x[1])

    # Display the digits on the frame. Used for debugging and for bragging.
    #display(frame, velocity_roi, altitude_roi, velocity_list, altitude_list)

    # Convert the velocity and altitude list to numbers.
    cur_velocity = digit_list_to_number(velocity_list)
    cur_altitude = digit_list_to_number(altitude_list)

    cur_altitude = decimal_point_conversion(cur_altitude)

    return cur_velocity, cur_altitude



# Returns the frame dimensions(width, height).
def get_frame_size(capture):
    return capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)




# Skip frames without telemetry based on the -T command line option.
def forward_to_launch(cap, time_before_launch):
    number_of_frames = int(cap.get(cv2.CAP_PROP_FPS) * time_before_launch)
    cap.set(cv2.CAP_PROP_POS_FRAMES, number_of_frames)





# Get two ROIs of the altitude and the velocity.
def crop_frame(frame, frame_height, frame_width, crop_points):
    return frame[int(crop_points[0] * frame_height): int(crop_points[1] * frame_height),
           int(crop_points[2] * frame_width): int(crop_points[3] * frame_width)]




# Get two threshed ROIs of the altitude and the velocity.
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
    cur_time = dt
    prev_velocity = 0
    prev_time = 0
    prev_altitude = 0
    data_frq = 1
    start_measurement = False
    frame_index = 0

    # Get frames dimensions.
    frame_width, frame_height = get_frame_size(cap)

    # Skip frames without telemetry based on the -T command line option.
    forward_to_launch(cap, launch_time)

    while cap.isOpened():
        #print(round_to_n_digits(frame_index / cap.get(cv2.CAP_PROP_FPS), 2), end='\r')

        # Read frame from capture.
        _, frame = cap.read()

        if frame is None:
            break

        # Add 1 to frame counter/uid.
        frame_index += 1

        # Skip this frame if the rocket hasn't launch yet.
        if start_measurement and frame_index % data_frq != 0:
            cur_time += dt
            continue

        # After the rocket launched add dt (time between frames) to the time since launch variable.
        if start_measurement:
            cur_time += dt

        # Crop frame to get only the altitude and velocity part of the frame.
        altitude_roi, velocity_roi = extract_frames(frame, frame_height, frame_width)

        # Get numeric values of velocity and altitude from the current frames.
        cur_velocity, cur_altitude = calculate_numbers(frame, templates, velocity_roi, altitude_roi)

        # Check if the values make sense.
        if cur_velocity is None:
            continue

        # Convert velocity from km/h to m/s.
        cur_velocity /= KMH_CONVERSION

        # Modify data to pass validity check if stream doesn't starts after the launch.
        if cur_velocity != 0 and cur_time == dt:
            prev_velocity = cur_velocity
            prev_altitude = cur_altitude
            prev_time = cur_time - dt

        # Make sure values make sense.
        if (not start_measurement and cur_velocity == 0) or not check_result(cur_time - prev_time,
                                                                            cur_velocity, prev_velocity,
                                                                            cur_altitude, prev_altitude):
            continue

        # If this frame is that contains valid telemetry, start counting the time.
        if not start_measurement:
            start_measurement = True

        dict_data = OrderedDict([
            ('time', round_to_n_digits(cur_time, 2)),
            ('velocity', round_to_n_digits(cur_velocity, 2)),
            ('altitude', round_to_n_digits(cur_altitude, 2))])

        # Write data to output file.
        file.write(json.dumps(dict_data)+'\n')
        file.flush()
        # json.dump(dict_data, file)

        # Send data to stdout.
        print(json.dumps(OrderedDict([('uid', frame_index), ('timestamp', datetime.datetime.utcnow().isoformat()+'Z'), ('data', dict_data)])))
        #print(output_data(frame_index, cur_time, current_velocity, current_altitude))

        # Update time, velocity and altitude values.
        prev_altitude = cur_altitude
        prev_velocity = cur_velocity
        prev_time = cur_time

    # Write the last data left to the disk.
    file.write(json.dumps(dict_data)+'\n')
    file.flush()
    # json.dump(dict_data, file)



def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Create graphs for SpaceX's launch videos.")
    parser.add_argument('-c', '--capture', action='store', dest='capture_path',
                        help='Path (url or local) of the desired video')
    parser.add_argument('-d', '--destination', action='store', dest='destination_path', default='telemetry.json',
                        help='Path to the file that will contain the output')
    parser.add_argument('-t', '--templates', action='store', dest='templates_path', default='Images',
                        help='Path to the folder that cotains the template images')
    parser.add_argument('-T', '--time', action='store', dest='launch_time', default=0,
                        help='Time delay from the beginning of the video to the time of the launch (in seconds)')

    args = parser.parse_args()

    # Use Livestreamer if the input is a url.
    if args.capture_path.startswith('www.youtube.com') or args.capture_path.startswith('http'):
        streams = livestreamer.streams(args.capture_path)
        capture = streams['1080'].url
    else:
        capture = args.capture_path

    # Notify the user if he/she tries to override a file.
    if isfile(args.destination_path):
        if input('{} already exists. Are you sure you want to overwrite it? [y/n]: '.format(args.destination_path)) != 'y':
            exit(1)

    # Open output files.
    file = open(args.destination_path, 'wt')

    # Load digits' templates from the disk.
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
