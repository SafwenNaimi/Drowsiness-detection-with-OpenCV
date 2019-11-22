# Drowsiness-detection-with-OpenCV
A computer vision system that can automatically detect driver drowsiness in a real-time video stream 
and then play an alarm if the driver appears to be drowsy.
It is build using OpenCv and Python 
We are interested in the region of the eye and the eye_aspect_ratio function which is used to compute the ratio of distances between 
the vertical eye landmarks and the distances between the horizontal eye landmarks.
The return value of the eye aspect ratio will be approximately constant when the eye is open.
The value will then rapid decrease towards zero during a blink.
If the eye is closed, the eye aspect ratio will again remain approximately constant, 
but will be much smaller than the ratio when the eye is open.





