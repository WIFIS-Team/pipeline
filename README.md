WIFIS pipeline

The following python packages must be installed:
Astropy >= 1.2
NumPy	>= 1.11.2
SciPy 	>= 0.18.0
pyopencl - for OpenCL accelerated tasks - >= 2015.2
matplotlib - for plotting >= 1.5.2
cPickle >= 2.3

The versions provide the release for which the pipeline was tested on. Your mileage may very with earlier versions.

Currently, the following must be copied/linked to the directory where to be run:
- all python scripts found under ~base~/core
- the opencl_code directory found under ~base~/core
- the external_data directory found under ~base~
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
