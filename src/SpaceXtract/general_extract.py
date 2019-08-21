import cv2
import numpy as np
from math import fabs
import os




class BaseExtract:
    """
    Extract data from frames when the location of the data is known and is constant
    """


    def __init__(self, image_dict):
        """
        :param image_dict: A dictionary that describes the templates in the image

         ------
         Format
         ------
         {
        "key1": [
            [top, bottom, left, right],
            ["path_to_template1.png", "path_to_template2.png", ...],
            threshold,
            [length1, length2, ...]
        ],
        ...
        }

        Note:
        1. [top, bottom, left, right] - ratios in the image.
        For example: [0, 0.5, 0, 0.5] is the top left quarter of the frame.


        2. [length1, length2, ...] - possible number of templates in the image.
        For example: For a clock with a 4 digit display (00:00-23:59) the list will be [4].
        """

        self.image_dict = image_dict

        # Load images to image_dict
        self.read_images_from_dict()




    def read_images_from_dict(self):
        """
        Replace the paths in image_dict with the templates(images) they point to
        """
        
        # A list of black and white versions of the templates
        for key in self.image_dict:
            img_lst = []
        
            for path in self.image_dict[key][1]:
                assert os.path.exists(path)
                img_lst.append(self.prepare_frame(cv2.imread(path), [0, 1, 0, 1]))

            # Replace path list with template list
            self.image_dict[key][1] = img_lst




    @staticmethod
    def ratio_to_pixel(points, shape):
        """
        Get the pixel values ([top, bottom, left, right]) of a template in location
        'points' in a frame with size 'shape'.

        Note: The bottom and right values are one more than the real values to allow easier slicing


        Example:

        [0.2, 0.8, 0.5, 1] in a 1080x1920 frame -> [216, 864, 960, 1920]

        :param points: [top, bottom, left, right] ratios in the frame
        :param shape: Frame dimensions (rows, columns)
        :return: [top, bottom, left, right] locations in the frame
        """
        return [
            int(points[0] * shape[0]),
            int(points[1] * shape[0]),
            int(points[2] * shape[1]),
            int(points[3] * shape[1])
        ]




    @staticmethod
    def pixel_to_ratio(points, shape):
        """
        Get the absolute location [top, bottom, left, right]
        of a template in ratio 'points' in a frame with size 'shape'.


        Example:

        [216, 864, 960, 1920] in a 1080x1920 frame -> [0.2, 0.8, 0.5, 1]

        :param points: [top, bottom, left, right] ratios in the frame
        :param shape: Frame dimensions (rows, columns)
        :return: [top, bottom, left, right] ratios in the frame
        """
        return [
            points[0] / shape[0],
            points[1] / shape[0],
            points[2] / shape[1],
            points[3] / shape[1]
        ]




    @staticmethod
    def crop_frame(frame, crop_points):
        """
        Returns the part of the frame imposed by 'crop_points'.

        :param frame: A frame
        :param crop_points: The 4 vertices of the rectangle [top, bottom, left, right].
        :return: Part of the frame imposed by crop_points.
        """
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        return frame[
             int(crop_points[0] * frame_height): int(crop_points[1] * frame_height),
             int(crop_points[2] * frame_width): int(crop_points[3] * frame_width)
        ]




    def prepare_frame(self, frame, points, thresh=150):
        """
        Returns the frame in black/white and cropped.

        :param frame: A frame
        :param points: [top, bottom, left, right] ratios used to crop the image
        :param thresh: Threshold for threshing process (0-255)
        :return:
        """
        roi = self.crop_frame(frame, points)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)

        return roi



    def image_to_location(self, frame, template, thresh=0, crop_range=None):
        """
        Get the most probable location of template in frame.

        :param frame: A frame.
        :param template: An image to search in 'frame'.
        :param thresh: The minimum probability required to accept template as potentially found.
        :param crop_range: Limit the area in 'frame' to look for 'template'.
        :return: A tuple (left, top) of the most probable location of template in frame.
        """

        # Set crop range default value
        if crop_range is None:
            crop_range = [0, 1, 0, 1]

        frame = self.prepare_frame(frame, crop_range)
                
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= thresh)

        point = None
        max_p = 0

        for pt in zip(*loc[::-1]):
            probability = res[pt[1]][pt[0]]
            if probability > max_p:
                point = pt
                max_p = probability

        return point



    @staticmethod
    def exists(frame, template, thresh):
        """
        Returns True if 'template' is in 'frame' with probability of at least 'thresh'
        :param frame: A frame
        :param template: An image to search in 'frame'.
        :param thresh: The minimum probability required to accept template.
        :return: If template is in frame
        """

        digit_res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(digit_res >= thresh)

        if len(loc[-1]) == 0:
            return False

        for pt in zip(*loc[::-1]):
            if digit_res[pt[1]][pt[0]] == 1:
                return False

        return True



    @staticmethod
    def remove_duplicates(lst, min_distance=10):
        """
        Remove digits found multiple times by the OCR algorithm.

        :param lst: List of digits, their positions and probability (Of the digit detected by the OCR algorithm)
        :param min_distance: Minimum distance that distinguishes two different characters
        :return: lst
        """
        for i in lst:
            for j in lst:
                if i != j:
                    # Remove values too close together (less than 'min_distance' pixels apart).
                    if fabs(i[1] - j[1]) < min_distance:
                        if j[2] < i[2]:
                            lst.remove(j)
                        else:
                            lst.remove(i)
                            break

        return lst





    @staticmethod
    def most_probably_template(image, templates):
        """
        Get the index of the template(in the templates list) which is most likely to be in the image.


        :param image: Image that contain the template
        :param templates: A list of templates to search in image
        :return: the index (in templates) which has the highest probability of being in  image
        """
        probability_list = []

        for template in templates:
            res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            probability_list.append(float(np.max(res)))

        return probability_list.index(max(probability_list))




    def get_template_index(self, frame, key):
        """
        Get the index of the template(in the templates list) which is most likely to be in the image.

        :param frame: A frame
        :param key: key in self.image_dict that is analysed
        :return:
        """

        roi = self.prepare_frame(frame, self.image_dict[key][0])
        templates = self.image_dict[key][1]
        return self.most_probably_template(roi, templates)


    @staticmethod
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




    @staticmethod
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



    def get_template_distance(self, pos_list):
        return [pos_list[i+1] - pos_list[i] for i in range(len(pos_list)-1)]





    def get_template_distance(self, pos_list):
        return [pos_list[i+1] - pos_list[i] for i in range(len(pos_list)-1)]

    
    
    def decimal_point_conversion(self, lst):
        return False
    


    def image_to_number(self, image, templates, threshold, lengths, decimal_func=None):
        """
        Returns the number in 'image'.


        :param image: An image of the field.
        :param templates: Templates of the digits in the field.
        :param threshold: Minimum threshold to detect the digits.
        :param lengths: Expected length of the field. (Number of digits).
        :return: Gap, The number in the image.
        """
        
        if decimal_func is None:
            decimal_func = self.decimal_point_conversion

        # Make sure the digits were extracted properly.
        number_list = self.image_to_digit_list(image, templates, threshold)

        # Remove duplicate values.
        number_list = self.remove_duplicates(self.remove_duplicates(number_list))

        if len(number_list) not in lengths:
            return False, None

        # Sort the digits of the velocity and altitude values by position on the frame.
        number_list.sort(key=lambda x: x[1])
        
        gap = decimal_func([x[1] for x in number_list])
        number = self.digit_list_to_number(number_list)

        return gap, number




    def extract_number(self, frame, key, decimal_func=None):
        """
        Get the number of 'key' in 'frame'

        :param frame: A frame
        :param key: The name(a key in 'image_dict') of the field to extract from 'frame'
        :return: Gap, The number in the frame
        """
        roi = self.prepare_frame(frame, self.image_dict[key][0])
        gap, number = self.image_to_number(roi, self.image_dict[key][1], self.image_dict[key][2], self.image_dict[key][3], decimal_func)

        return gap, number












class RelativeExtract(BaseExtract):
    """
    Extract data from frames when the location of the data is known and constant RELATIVE to an
    image in the frame.

    parent: BaseExtract
    """


    def __init__(self, image_dict, anchor_range=None, anchor_moving=False):
        """
        :param image_dict: Same as BaseExtract
        :param anchor_range: An area the anchor is expected to be in
        :param anchor_moving: Is the anchor moving between frames?
        """
        BaseExtract.__init__(self, image_dict)

        # Set default value to 'anchor_range'
        if anchor_range is None:
            anchor_range = [0, 1, 0, 1]

        self.anchor_image = image_dict['anchor'][1][0]
        self.anchor_range = anchor_range
        self.anchor_thresh = image_dict['anchor'][2]

        self.anchor_location = None

        self.anchor_moving = anchor_moving
        self.relative_image_dict = self.image_dict
        self.image_dict = None



    def prepare_image_dict(self, frame):
        """
        Prepare image_dict for frame

        :param frame: A frame
        :return: If frame can be used for further analysis
        """
        if self.anchor_moving or self.anchor_location is None:
            if not self.search_anchor(frame):
                return False
            self.create_image_dict()

        return self.image_dict is not None



    def search_anchor(self, frame):
        """
        Search anchor in 'frame' and set its location (self.anchor_location) accordingly
        :param frame: A frame
        :return: If anchor has been found
        """

        anchor_pixel_location = self.image_to_location(frame, self.anchor_image, self.anchor_thresh, self.anchor_range)

        if anchor_pixel_location is None:
            return False

        anchor_top_bottom_left_right = [
                anchor_pixel_location[1],
                anchor_pixel_location[1] + self.anchor_image.shape[0],
                anchor_pixel_location[0],
                anchor_pixel_location[0] + self.anchor_image.shape[1]
        ]

        anchor_ratio = np.array(self.pixel_to_ratio(anchor_top_bottom_left_right, frame.shape))

        crop_ratio = np.array([
            self.anchor_range[0],
            self.anchor_range[0],
            self.anchor_range[2],
            self.anchor_range[2]])

        self.anchor_location = list(anchor_ratio + crop_ratio)

        return True



    def relative_to_abs(self, points):
        return [
            self.anchor_location[0] + points[0],
            self.anchor_location[0] + points[1],
            self.anchor_location[2] + points[2],
            self.anchor_location[2] + points[3],
        ]


    def create_image_dict(self):
        """
        Convert the dictionary of relative location (to the anchor)
        to an absolute location dictionary (without the anchor)
        """
        self.image_dict = {}

        for key in self.relative_image_dict:
            if key != 'anchor':
                self.image_dict[key] = self.relative_image_dict[key]
                self.image_dict[key][0] = self.relative_to_abs(self.relative_image_dict[key][0])


    def get_template_index(self, frame, key):
        """
        Get the index of the template(in the templates list) which is most likely to be in the image.

        :param frame:
        :param key:
        :return:
        """
        if not self.prepare_image_dict(frame):
            return None

        return super(RelativeExtract, self).get_template_index(frame, key)



    def extract_number(self, frame, key, decimal_func=None):
        """
        Get the number of 'key' in 'frame'

        :param frame: A frame
        :param key: The name(a key in 'image_dict') of the field to extract from 'frame'
        :return: The number in the frame
        """
        if not self.prepare_image_dict(frame):
            return None
        
        return super(RelativeExtract, self).extract_number(frame, key, decimal_func)
