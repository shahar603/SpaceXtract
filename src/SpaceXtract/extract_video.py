import re
import cv2
import os
import streamlink


def youtube_url_validation(url):
    """
    Check if cap_path is a URL of a YouTube video
    :param url: The checked string
    :return: is url a valid url
    """
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    youtube_regex_match = re.match(youtube_regex, url)

    return youtube_regex_match is not None



def get_capture(cap_path):
    """
    Get OpenCV capture
    :param cap_path:
    :return:
    """

    # Check if cap_path is a URL of a YouTube video
    if youtube_url_validation(cap_path):
        return get_capture_from_url(cap_path, '1080p')

    elif os.path.isfile(cap_path):
        return cv2.VideoCapture(cap_path)

    return None


def is_live(cap):
    """
    Returns True if the capture is live and False otherwise
    :param cap: An OpenCV capture
    :return: True if the capture is live and False otherwise
    """
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < 0



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




def get_capture_from_url(youtube_url, res):
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