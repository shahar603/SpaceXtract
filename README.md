# SpaceXtract
Extraction and analysis of telemetry from SpaceX webcasts.
This module is built for Python 3 (I tested it on Python 3.5.1 32 bit version on Windows 10). You'll need [OpenCV](http://opencv.org/), [NumPy](http://www.numpy.org/), [Streamlink](https://streamlink.github.io/) and [FFMpeg](https://ffmpeg.org/)


Installing the required modules
-----------------------------

All the required modules can be installed using pip in the following manner:
```
pip install numpy
pip install opencv-python
pip install streamlink
```
You will need [FFMpeg](https://ffmpeg.org/) to be installed and to be in ```PATH```


extract.py
=========
[extract.py](https://github.com/shahar603/SpaceX/blob/master/extract.py) is a Python module that allows anyone with a little knowledge of OpenCV to be able to write a program that captures data from SpaceX's webcasts. Live or not, using a local video file or just a link to YouTube.



Importing the module
--------------------
Put the extract.py script and the Templates folder in the same directory as your script.

To import it to your script, add this line:
```
import extract
```

Currently, the extract module cannot be installed with any tool (To my very limited knowledge).
To get the module, download [extract.py](https://github.com/shahar603/SpaceX/blob/master/extract.py) and the [Templates](https://github.com/shahar603/SpaceX/tree/master/Templates) directory from this repository. In the future I plan to make it pip installable.

Documentation
--------------------
The module contains quite a lot of functions, but only a few are made for the user.
Here are details about the useful functions for the user from the documentation.

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
  
rtnd(number, n)
    Round number to a max of n digits after the decimal point
    :param number: Given number
    :param n: Requested number of digits after the decimal points
    :return: number with a max of n digits after the decimal point

is_live(cap):
    Returns True if the capture is live and False otherwise
    :param cap: An OpenCV capture
    :return: True if the capture is live and False otherwise
```


Example of use
--------------------
This is a script that outputs the time, velocity and altitude values of the Inmarsat 5 F4 launch
```python
import extract
import cv2

# Get OpenCV capture of the video
cap = extract.get_capture('https://www.youtube.com/watch?v=ynMYE64IEKs', '1080p')

# Exit if cannot get capture
if cap is None:
    exit(1)

# Move capture to launch. If live this line does nothing.
extract.skip_to_launch(cap)

# Read the first frame
_, frame = cap.read()

# While the video hasn't finished
while frame is not None:
    # Calculate the time, velocity and alitutde values from the frame
    # If can't calculate values, returns (None, None, None)
    time, velocity, altitude = extract.extract_telemetry(frame)

    # If values are valid, print them
    if time is not None:
        print(time, velocity, altitude)
        
    # Read the next frame
    _, frame = cap.read()
```
This script can be downloaded from [here](https://github.com/shahar603/SpaceX/blob/master/example.py).
If you just want a program that does the job without having to program it, you can download it form [here](https://github.com/shahar603/SpaceX/blob/master/get_telemetry.py). It
extracts the data, check it's valid and write it to a JSON file. Explanations about its usage can be found below in the section titled "get_telemetry.py".

- **I Highly recommend using 1080p, 720p is supported but is pretty dull. From my tests, 1080p is correct 98% of the time while 720p is less than 60%**

- **The final Python package might not be compadible with older versions of this module**




Plans for the future
--------------------
* Make the module a Python package
* Better names and documentation (New names for functions, variables, tables and more). I'm open for suggestions.
* Better support for 720p video
* Moving some settings to a configuration files instread of being hard coded
* (Maybe?) Support for other space streams (for example Blue Origin's)

### Feedback is very welcome

get_telemetry.py
=====================
This script extract the telemetry data from SpaceX's webcast.
This scripts uses the [extract.py](https://github.com/shahar603/SpaceX/blob/master/extract.py) module.

Command line arguments
-------------------
Output of ```python get_telemetry.py --help```

```
usage: get_telemetry.py [-h] [-c CAPTURE_PATH] [-d DESTINATION_PATH]
                        [-T LAUNCH_TIME] [-o]

Create graphs for SpaceX's launch videos.

optional arguments:
  -h, --help            show this help message and exit
  -c CAPTURE_PATH, --capture CAPTURE_PATH
                        Path (url or local) of the desired video
  -d DESTINATION_PATH, --destination DESTINATION_PATH
                        Path to the file that will contain the output
  -T LAUNCH_TIME, --time LAUNCH_TIME
                        Time from launch of the video to the time of the
                        launch (in seconds). If not given and not live, the
                        capture is set to the launch. If live, the capture
                        isn't be affected
  -o                    If given results will be printed to stdout
```


Examples of use
----------------------

```
python get_telemetry.py -c "JCSAT-14 Hosted Webcast.mp4" -d "JCSAT-14.json"
```
* This command will write the data from the JCSAT-14 webcast local file to the file "JCSAT-14.json".

```
python get_telemetry.py -c https://www.youtube.com/watch?v=L0bMeDj76ig -d "JCSAT-14.json"
```
* This command will write the data from the JCSAT-14 webcast's YouTube video to the file "JCSAT-14.json".


### WARNING: The program WILL OVERRIDE the output file. Be careful when you run it.

