WIFIS pipeline

The following python packages must be installed:
Astropy
NumPy
SciPy
pyopencl - for OpenCL accelerated tasks
matplotlib - for plotting

Main scripts to edit and run:
wifisCalDetLin.py
wifisCalDark.py
wifisCalWave.py
wifisCalFlat.py
wifisCalObs.py

These main scripts will call the secondary scripts or opencl codes.

Currently, only wifisCalDetLin.py and wifisCalDark.py are fully functional. wifisCalWave.py is mostly complete and is functional.
