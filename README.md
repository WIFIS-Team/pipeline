==============
WIFIS pipeline
==============

*This is a in-development branch for the WIFIS pipeline that attempts to catch up to python 3 and the latest package updates*


This package requires and has has been tested on:
Astropy = 4.0
NumPy	= 1.18.1
SciPy 	= 1.3.2
pyopencl - for OpenCL accelerated tasks - = 2019.1.2
matplotlib - for plotting = 3.1.1
colorama = 0.4.3

The versions provided are the releases for which the pipeline was tested on. Your mileage may very with earlier versions.

To setup:
add the core directory containing the core modules to your PYTHONPATH environment variable.

The code is largely designed such that: the main scripts are copied/linked to the directory where to be run, along with the configuration file:
- the python scripts found under ~base~/scripts
where ~base~ is the folder where the pipeline is installed.

Main scripts to run, editing should (mostly) not be needed:
wifisCalDetLin.py
wifisCalDark.py
wifisCalWave.py
wifisCalFlat.py
wifisCalObs.py
wifisCalSpatialCor.py
wifisCalObs.py

These main scripts will call the secondary scripts, functions, or opencl codes as needed.

