import cv2
import numpy as np
from os import sep
from math import fabs
import streamlink

thresh_dict = {
    1080: (0.85, 0.7),
    720:  (0.7, 0.55)
}

KMH_CONVERSION = 3.6  # m/s

#                   [top]     [bottom]       [left]       [right]
rects = {
    'altitude' : [0.185185185, 0.231481481,  0.911458333, 0.963541667],
    'velocity' : [0.185185185, 0.231481481,  0.807291667, 0.885416667],
    'point' :    [0.215740741, 0.22037037,   0.94375,     0.946875],
    'unit' :     [0.234259259, 0.252777778,  0.846354167, 0.875],
    'sign' :    [0.037037037, 0.0564814815, 0.799479167, 0.809895833],
    'time' :     [0.029629629, 0.0574074074, 0.832291667, 0.911041667]
}

current_dir = None


def rtnd(number, n):
    """
    Round number to a max of n digits after the decimal point
    :param number: Given number
    :param n: Requested number of digits after the decimal points
    :return: number with a max of n digits after the decimal point
    """
    return int(number * 10 ** n) / 10 ** n


def skip_from_launch(cap, time):
    """
    Move the capture to T+time (time can be negative) and returns the frame index.
    :param cap: OpenCV capture
    :param time: delta time from launch to skip to
    :return: index of requested frame
    """
    number_of_frames = int(cap.get(cv2.CAP_PROP_FPS) * time) + skip_to_launch(cap)

    number_of_frames = max(number_of_frames, 0)
    number_of_frames = min(number_of_frames, cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, number_of_frames)

    return number_of_frames


def remove_duplicates(lst):
    """
    Remove digits found twice or more by the OCR algorithm.
    :param lst: List of digits, their positions and probability (Of the digit detected by the OCR algorithm)
    :return: lst 
    """
    for i in lst:
        for j in lst:
            if i != j:
                # Remove values too close together (less than 10 pixels apart).
                if fabs(i[1] - j[1]) < 10:
                    if j[2] < i[2]:
                        lst.remove(j)
                    else:
                        lst.remove(i)
                        break


    return lst


def is_live(cap):
    """
    Returns True if the capture is live and False otherwise
    :param cap: An OpenCV capture
    :return: True if the capture is live and False otherwise
    """
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < 0


def digit_list_to_number(digit_list):
    """
    Convert a list of digits to a number.
    :param digit_list: The list of digits to convert
    :return: a number from the elements of the list
    """
    number = 0

    # Convert the list of digits to a number.
    for digit, _, p in digit_list:
        number = 10 * number + digit

    return number


def image_to_digit_list(image, digit_templates, thresh):
    """
    Convert an image to a list of digits.
    :param image: The part of the image containing the number.
    :param digit_templates: Images of all the digits.
    :param thresh: The threshold required to detect an image.
    :return: a list of digits with data about the probability they were found in the image and their position.
    """
    # Initialize variables.
    digit_list = []
    digit = 0

    # Convert the values from the 'height' and 'velocity' images.
    for digit_image in digit_templates:
        # Get a matrix of values containing the probability the pixel is the top-left part of the template.
        digit_res = cv2.matchTemplate(image, digit_image, cv2.TM_CCOEFF_NORMED)

        # Get a list of all the pixels that have a probability >= to the thresh.
        loc = np.where(digit_res >= thresh)

        # Create a list that contains the x position and the digit.
        for pt in zip(*loc[::-1]):
            digit_list.append((digit, pt[0], digit_res[pt[1]][pt[0]]))

        digit += 1

    return digit_list


def exists(image, template, thresh):
    """
    Returns True if template is in Image with probability of at least thresh
    :param image: 
    :param template: 
    :param thresh: 
    :return: 
    """
    digit_res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(digit_res >= thresh)

    if len(loc[-1]) == 0:
        return False

    for pt in zip(*loc[::-1]):
        if digit_res[pt[1]][pt[0]] == 1:
            return False

    return True


def velocity_to_ms(velocity, image, template, thresh):
    """
    Convert velocity to m/s from the unit in the stream
    :param velocity: The velocity in the stream. 
    :param image: An image of the unit of the velocity.
    :param template: template of the unit (km/h).
    :param thresh: Minimum threshold to detect the unit.
    :return: 
    """
    if velocity is None:
        return None

    if exists(image, template, 0.6):
        return velocity / KMH_CONVERSION

    return velocity


def skip_to_launch(cap):
    """
    Move cap to the frame before the launch
    :param cap: An OpenCV capture of the launch.
    :return: the index of first frame at T+00:00:00
    """
    initialize(1080)

    left = 0
    right = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, int((right+left)/2))

    while right > left+1:
        _, frame = cap.read()
        image = crop(frame, rects['sign'])

        if exists(image, sign_template, thresh_dict[frame.shape[0]][1]):
            left = int((right+left)/2)
        else:
            right = int((right+left)/2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, int((right + left) / 2))

    cap.set(cv2.CAP_PROP_POS_FRAMES, left)

    return left


def decimal_point_conversion(digit_pos_list, altitude):
    """
    Change the altitude value according to the position of the decimal point.
    :param digit_pos_list: a list of the digits with their position.
    :param altitude: Value of altitude
    :return: The altitude after decimal digit conversion.
    """
    if digit_pos_list[2][1]-digit_pos_list[1][1] > templates[0].shape[1]*1.5:
        return altitude / 10

    return altitude


def image_to_number(image, templates, threshold, length):
    """
    Returns the number in the Image
    :param image: An image of the field.
    :param templates: Templates of the digits in the field.
    :param threshold: Minimum threshold to detect the digits.
    :param length: Expected length of the field. (Number of digits).
    :return: The number in the image.
    """
    number_list = image_to_digit_list(image, templates, threshold)

    # Remove duplicate values.
    number_list = remove_duplicates(number_list)

    # Make sure the digits were extracted properly.
    if len(number_list) != length:
        return None

    # Sort the digits of the velocity and altitude values by position on the frame.
    number_list.sort(key=lambda x: x[1])

    # Convert the velocity and altitude list to numbers.
    return digit_list_to_number(number_list)


def time_to_seconds(time):
    """
    Convert the time to seconds
    :param time: A number that represent the time. ex. 011731 --> 1 hour, 17 minutes and 31 seconds
    :return: The number in seconds.
    """
    seconds = time % 100
    minutes = int(time / 100) % 100
    hours = int(time / 10000)
    return seconds + 60*minutes + 60*60*hours


def crop_frame(frame, crop_points):
    """
    Returns part of the frame imposed by crop_points.
    :param frame: A frame
    :param crop_points: The 4 vertices of the rectangle.
    :return: Part of the frame imposed by crop_points.
    """
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    return frame[int(crop_points[0] * frame_height): int(crop_points[1] * frame_height),
           int(crop_points[2] * frame_width): int(crop_points[3] * frame_width)]


def initialize(res):
    """
    Loads all the templates from the files
    :param res: Resolution of the video.
    """
    global current_dir

    if res == current_dir:
        return

    global templates, unit_template, sign_template, time_templates

    path = 'Templates' + sep + str(res)

    templates = [cv2.imread(path + sep + str(digit) + '.jpg', 0) for digit in range(10)]
    time_templates = [cv2.imread(path + sep + 'Time' + sep + str(digit) + '.jpg', 0) for digit in range(10)]
    unit_template = cv2.imread(path + sep + 'kmh.jpg', 0)
    sign_template = cv2.imread(path + sep + 'minus.jpg', 0)

    _, unit_template = cv2.threshold(unit_template, 150, 255, cv2.THRESH_BINARY)
    _, sign_template = cv2.threshold(sign_template, 150, 255, cv2.THRESH_BINARY)

    for i in range(10):
        _, templates[i] = cv2.threshold(templates[i], 150, 255, cv2.THRESH_BINARY)
        _, time_templates[i] = cv2.threshold(time_templates[i], 150, 255, cv2.THRESH_BINARY)

    current_dir = res


def crop(frame, points, p = 150):
    """
    Returns the frame cropped and converted to black and white.
    :param frame: 
    :param points: 
    :param p: 
    :return: 
    """
    roi = crop_frame(frame, points)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, p, 255, cv2.THRESH_BINARY)

    return roi


def get_url(youtube_url, res):
    """
    Gets a direct URL to the video.
    :param youtube_url: The url of the video.
    :param res: The resolution of the video.
    :return: a string of the direct URL of the youtube video.
    """

    streams = streamlink.streams(youtube_url)
    if res not in streams:
        return None
    elif type(streams[res]) != streamlink.stream.ffmpegmux.MuxedStream:
        return streams[res].url
    return streams[res].substreams[0].url


def get_capture(youtube_url, res):
    """
    Get an OpenCV capture of a YouTube video.
    :param youtube_url: A url of the video
    :param res: The resolution of the video.
    :return: An OpenCV capture of the video.
    """
    for i in range(3):
        try:
            url = get_url(youtube_url, res)

            if url is None:
                continue

            cap = cv2.VideoCapture(url)

            if cap is not None:
                return cap
        except:
            pass

    return None


def calc_time(frame):
    """
    Get the time in the frame.
    :param frame: Frame from the launch.
    :return: The time in the frame.
    """
    res = frame.shape[0]
    initialize(res)
    time_roi = crop(frame, rects['time'])

    time = image_to_number(time_roi, time_templates, thresh_dict[res][0], 6)

    if time is None:
        return None

    if exists(crop(frame, rects['sign']), sign_template, thresh_dict[res][1]):
        return -time_to_seconds(time)
    return time_to_seconds(time)


def calc_velocity(frame):
    """
    Get the velocity in the frame.
    :param frame: Frame from the launch.
    :return: The velocity in the frame.
    """
    res = frame.shape[0]
    initialize(res)
    velocity_roi = crop(frame, rects['velocity'])
    velocity = image_to_number(velocity_roi, templates, thresh_dict[res][0], 5)

    if velocity is None:
        return None

    velocity = velocity_to_ms(velocity, crop(frame, rects['unit']), unit_template, thresh_dict[res][1])

    return rtnd(velocity, 3)


def calc_altitude(frame):
    """
    Get the altitude in the frame.
    :param frame: Frame from the launch.
    :return: The altitude in the frame.
    """
    res = frame.shape[0]
    initialize(res)
    altitude_roi = crop(frame, rects['altitude'])
    thresh = thresh_dict[res][0]
    altitude = image_to_number(altitude_roi, templates, thresh, 3)

    if altitude is None:
        return None

    lst = image_to_digit_list(altitude_roi, templates, thresh)
    lst = remove_duplicates(lst)
    lst.sort(key=lambda x: x[1])

    return rtnd(decimal_point_conversion(lst, altitude), 3)


def extract_telemetry(frame):
    """
    Calculate rocket telemetry from the launch
    :param frame: A frame
    :return: A tuple: (Time, Velocity, Altitude)
    """
    time, velocity, altitude = calc_time(frame), calc_velocity(frame), calc_altitude(frame)

    if time is not None and velocity is not None and altitude is not None:
        return time, velocity, altitude

    return None, None, None
