# SpaceX
Extraction and analysis of telemetry from SpaceX webcasts
Run this program using Python 3 (I tested it on Python 3.5.1 32bit version). You'll need [OpenCV](http://opencv.org/), [NumPy](http://www.numpy.org/) and [Livestreamer](http://docs.livestreamer.io/).

Example of use:
```
python get_telemetry.py -c "JCSAT-14 Hosted Webcast.mp4" -d "JCSAT-14.json" -T 1250 -r false
```
