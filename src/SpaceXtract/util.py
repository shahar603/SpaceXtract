import cv2
from SpaceXtract.general_extract import BaseExtract, RelativeExtract

import random


class Util():

    def __init__(self, extractor: BaseExtract):
        self.extractor = extractor


    def search_switch(self, cap, key, thresh=0.5):
        left = 0
        right = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, int((right + left) / 2))

        while right > left + 1:
            _, frame = cap.read()

            image = self.extractor.prepare_frame(frame, self.extractor.image_dict[key][0])

            if not self.extractor.exists(image, self.extractor.image_dict[key][1][0], thresh):
                left = int((right + left) / 2)
            else:
                right = int((right + left) / 2)

            cap.set(cv2.CAP_PROP_POS_FRAMES, int((right + left) / 2))

        cap.set(cv2.CAP_PROP_POS_FRAMES, left)

        return left



    def skip_from_launch(self, cap, key, time, thresh=None):
        """
        Move the capture to T+time (time can be negative) and returns the frame index.
        :param cap: OpenCV capture
        :param time: delta time from launch to skip to
        :return: index of requested frame
        """        
        if thresh is None:
            thresh = self.extractor.image_dict[key][2]

        number_of_frames = int(cap.get(cv2.CAP_PROP_FPS) * time) + self.search_switch(cap, key, thresh)

        number_of_frames = max(number_of_frames, 0)
        number_of_frames = min(number_of_frames, cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, number_of_frames)

        return number_of_frames



    def find_anchor(self, cap, start=0, end=1, maxiter=10):
        if not isinstance(self.extractor, RelativeExtract):
            return False

        original_location = cap.get(cv2.CAP_PROP_POS_FRAMES)

        for i in range(maxiter):
            pos = random.uniform(start, end)

            cap.set(cv2.CAP_PROP_POS_FRAMES,  pos*cap.get(cv2.CAP_PROP_FRAME_COUNT))
            _, frame = cap.read()

            if self.extractor.prepare_image_dict(frame):
                return True

        cap.set(cv2.CAP_PROP_POS_FRAMES, original_location)

        return False



    def get_template_distance(self, pos_list):
        return [pos_list[i+1] - pos_list[i] for i in range(len(pos_list)-1)]


    def decimal_point_conversion(self, digit_pos_list):
        """
        Change the altitude value according to the position of the decimal point.
        :param digit_pos_list: a list of the digits with their position.
        :param altitude: Value of altitude
        :return: The altitude after decimal digit conversion.
        """

        distances = self.get_template_distance(digit_pos_list)
        return distances[-2] > distances[-1] * 1.3
        
    def play_until_anchor_found(self, cap, session, interval=5):
        _, frame = cap.read()
        
        while not session.prepare_image_dict(frame):
            for i in range(interval*30):
                _, frame = cap.read()
                
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            
            cv2.imshow('frame', frame)
           
                



def rtnd(number, n):
    """
    Round number to a max of n digits after the decimal point
    :param number: Given number
    :param n: Requested number of digits after the decimal points
    :return: number with a max of n digits after the decimal point
    """
    return int(number * 10 ** n) / 10 ** n



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


def to_float(num):
    try:
        return float(num)
    except ValueError:
        return 0