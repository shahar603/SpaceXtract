# SpaceXtract

![SpaceXtract gif](https://github.com/shahar603/SpaceXtract/blob/master/docs/SpaceXtract.gif)


Extraction and analysis of telemetry from launch provders' webcasts (like SpaceX and RocketLab).
This module is built for Python 3. You'll need [OpenCV](http://opencv.org/), [NumPy](http://www.numpy.org/), [Streamlink](https://streamlink.github.io/) and [FFMpeg](https://ffmpeg.org/)




### Note: Previous versions of this software are available on [Old Extraction Scripts](https://github.com/shahar603/SpaceXtract/tree/88f255da4841f4b4015474b3e74bf8e7de1eb64e/Old%20Extraction%20Scripts). It may not be compatible with newer visualizations and analysis tools
 


Installing the required modules
==========================



All the required modules can be installed using pip in the following manner:

```
pip install -r requirements.txt
```

Or manualy by installing the individul modules:

```
pip install numpy
pip install opencv-python
pip install streamlink
pip install matplotlib
```

You will need [FFMpeg](https://ffmpeg.org/) to be installed and to be in ```PATH```


Usage
=========

To capture telemetry from SpaceX's webcasts clone this repository and run ```python get_telemetry_spacex.py``` with the webcast (video file).


Here's the output of the ```--help``` option:

```
usage: get_telemetry_spacex.py [-h] [-c CAPTURE_PATH] [-d DESTINATION_PATH]
                               [-T LAUNCH_TIME] [-o] [-f]

Extract telemetry for SpaceX's launch videos.

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
  -f                    Force override of output file
```




Extraction of telemetry from other sources
==============

```get_telemetry_spacex.py``` uses the [SpaceXtract](https://github.com/shahar603/SpaceXtract/tree/master/src/SpaceXtract) package.

[SpaceXtract](https://github.com/shahar603/SpaceXtract/tree/master/src/SpaceXtract) is a package that performes fast OCR by searching and parsing only the data the user needs.
To do that it uses JSON configuration files (their format is specified [below]()).

[SpaceXtract](https://github.com/shahar603/SpaceXtract/tree/master/src/SpaceXtract) uses the general_extract.py script to perform OCR.

general_extract.py contains two classes, ```BaseExtract``` and ```RelativeExtract```.


```RelativeExtract``` is a subclass of ```BaseExtract```, the main difference between the two is that BaseExtract performes
OCR on a fixed region of interest on screen. In contrast, RelativeExtract tracks the region of interest even when it moves and changes.




Configuration files
------------------

The configuration file is a JSON file which contains a dictionary (keys - string, value - a list of size 4) that tell BaseExtract and it's subclasses where to perform
OCR, what characters to search, how clear are the characters and how many characters to expect to find.

The format of the file is the following:

```javascript
{
    "field_1": [
        [top, bottom, left, right],
        
        ["path_to_template_1.png", "path_to_template_2.png", ...],
        
        threshold,
        
        [expected_length_1, expected_length_2, ...]
    ],
    
    "field_2" : [
        ...
    ],
    
    ...
    
}
```


* "field_1" - The name of the field to search.

* [top, bottom, left, right] - A list that specify the area to perform OCR. top, bottom, left and right
are in ratio to screen size.

For example: The list [0.1, 0.9, 0.4, 0.6] on a 1920x1080 image captures a rectangle with dimentions:

(Rectangle_Width, Rectangle_Height) = (Screen_Width * (right - left), Screen_Height * (bottom - top)) = (1920*(0.6-0.4), 1080*(0.9-0.1)) = (384, 864)

The location of the rectangle is specific to the Extractor class used and is specified [below]().


* ["path_to_template_1.png", "path_to_template_2.png", ...] - A list of pathes to templates (images) to look for in the image.
The images can be colored, but they are converted to grayscale and only prominent features in the image (like edges) are used to detect characters.


* threshold - Minimum confidence required to detect a character.


* [expected_length_1, expected_length_2, ...] - A list of optional lengths of the output. 




Rectangle Location
-----------------


## BaseExtractor

The (top, left) corner of the rectangle is the same as the (top, left) value specified in the configuration file.

## RelativeExtractor

RelativeExtractor uses an *anchor*. A template whos location is used as a reference for the other fields.

If the anchor top left corner is (anchor_top, anchor_left), then the (top, left) corner of the rectangle RelativeExtract performes OCR in is (anchor_top+top, anchor_left+left).


The anchor is specified in the configuration file as follows:

```javascript
    "anchor": [
        null,
        [
            "path_to_the_anchor.png"
        ],
        threshold,
        []
    ] 
```



Usage
-----------

To use BaseExtract and RelativeExtract first import general_extract.py

```python
import general_extract
```

Then create a BaseExtract instance.

```python
session = general_extract.BaseExtract(configuration_file_content)
```

For a given OpenCV ```frame```, extract ```'my_field'``` using ```extract_number```:

```python
my_field = session.extract_number(frame, 'my_field')
```

```'my_field'``` is a number that contains the indecies of the templates as defined in ```configuration_file_content['my_field'][1]``` (Path to template list).
