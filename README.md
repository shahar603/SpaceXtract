# SpaceX
Extraction and analysis of telemetry from SpaceX webcasts.
Run this program using Python 3 (I tested it on Python 3.5.1 32bit version). You'll need [OpenCV](http://opencv.org/), [NumPy](http://www.numpy.org/), [Livestreamer](http://docs.livestreamer.io/), [Streamlink](https://streamlink.github.io/) and [FFMpeg](https://ffmpeg.org/)


extract.py
=========
extract.py is a Python module that allows anyone with a little knowledge of OpenCV to be able to analyse data from SpaceX webcast, Live, after launch or Offline.

Installing the required modules
===============================

All of these modules are can be installed using pip in the following manner:
```
pip install numpy
pip install opencv-python
pip install livestreamer
pip install streamlink
```

You will need [FFMpeg](https://ffmpeg.org/) to be installed and be in ```PATH```

Currently, the extract module cannot be installed with any tool (To my very limited knowledge).
To get the module, download [extract.py](https://github.com/shahar603/SpaceX/blob/master/extract.py) and the [Templates](https://github.com/shahar603/SpaceX/tree/master/Templates) directory from this repo. In the future I plan to make it pip installable.



Importing the module
===============================
Put the extract.py and the Templates folder in the same directpory as your script.
To import it to your script, add this line:
```
import extract.py
```


Documentation
===============================
The module contains quite a lot of functions, but only a few were made for the user.
Here is details about the useful function for the user from help(extract).

```
calc_altitude(frame)
  Get the altitude in the frame.
  :param frame: Frame from the launch.
  :return: The altitude in the frame.

calc_time(frame)
  Get the time in the frame.
  param frame: Frame from the launch.
  return: The time in the frame.

calc_velocity(frame)
  Get the velocity in the frame.
  :param frame: Frame from the launch.
  :return: The velocity in the frame.
        
skip_to_launch(cap) 
  Move cap to the first frame at T+00:00:00
  :param cap: An OpenCV capture of the launch.
  :return: the index of first frame at T+00:00:00
  
skip_from_launch(cap, time)
  Move the capture to T+time (time can be negative) and returns the frame index.
  :param cap: OpenCV capture
  :param time: delta time from launch to skip to
  :return: index of requested frame
  
extract_telemetry(frame)
  Get time, velocity and alitutde values in frame
  :param frame: Frame from the launch.
  :return: tuple of time, velocity and alitutde (In this order)
  
#### WARNING For res='1080p' the function takes between 10 seconds to 5 minutes. In addition to that it might fail (Return a None)
or raise an exception, if it does, try to run it again ####
get_capture(youtube_url, res)
  Get an OpenCV capture of a YouTube video.
  :param youtube_url: A url of the video
  :param res: The resolution of the video.
  :return: An OpenCV capture of the video.
```


Example of use
==============
This is a script that outputs the time, velocity and altitude values of the Inmarsat 5 F4 launch
```
import extract
import cv2

cap = extract.get_capture('https://www.youtube.com/watch?v=ynMYE64IEKs', '1080p')

if cap is None:
    exit(1)

extract.skip_to_launch(cap)

while cap.isOpened():
    _, frame = cap.read()

    time, velocity, altitude = extract.extract_telemetry(frame)

    if time is not None:
        print(time, velocity, altitude)
```


- **I Highly recommend using 1080p, 720p is supported but is pretty dull. From my tests, 1080p is correct 98% of the time while 720p is less than 60%**

- **The final Python package might not be compadible with older versions of this module**




Plans for the future
====================

* Make the module a Python package
* Better names and documentation (New names for functions, variables, tables and more). I'm open for suggestions.
* Better support for 720p video
* Moving some settings to a configuration files instread of being hard coded
* (Maybe?) Support for other space streams (for example Blue Origin's)

**Feedback is very welcome**


get_telemetry.py (Old)
----------
**-New version will be uploaded soon**

Example of use:
```
python get_telemetry.py -c "JCSAT-14 Hosted Webcast.mp4" -d "JCSAT-14.json" -T 1250 -r false
```
