WIFIS pipeline

The following python packages must be installed:
Astropy >= 1.2
NumPy	>= 1.11.2
SciPy 	>= 0.18.0
pyopencl - for OpenCL accelerated tasks - >= 2015.2
matplotlib - for plotting >= 1.5.2
cPickle >= 2.3

The versions provided are the releases for which the pipeline was tested on. Your mileage may very with earlier versions.

To setup:
add the core directory containing the core modules to your PYTHONPATH environment variable.

The code is largely designed such that: the main scripts are copied/linked to the directory where to be run:
- the python scripts found under ~base~/scripts
where ~base~ is the folder where the pipeline is installed.

Main scripts to edit and run:
wifisCalDetLin.py
wifisCalDark.py
wifisCalWave.py
wifisCalFlat.py
wifisCalObs.py
wifisCalSpatialCor.py
wifisCalObs.py

These main scripts will call the secondary scripts or opencl codes as needed.

The main scripts are a work in progress and may not work as intended at this point in time.
