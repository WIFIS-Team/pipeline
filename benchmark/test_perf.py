#import matplotlib
#matplotlib.use('tkagg')

import wifisIO
import matplotlib.pyplot as plt
import numpy as np
import wifisCombineData as combineData
import wifisNLCor as nlCor
import wifisGetSatInfo as satInfo
import wifisRefCor as refCor
import wifisSlices as slices
import wifisSpatialCor as spatialCor
import wifisWaveSol as waveSol
import wifisCreateCube as createCube
import os
import time

#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = ':1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#***********************************
outfile='benchmark_results.out'
nruns = 1
ncpuLst = [4]
#***********************************

savefile = open(outfile,'wb', 0)
tStart = time.time()

#test reference pixel corrections
data, inttime, hdr = wifisIO.readImgsFromFile('WIFIS_H2RG_SingleEnd_100KHz_18dB.47.1.1.fits')
print('benchmarking OpenCL portion')
print('testing channel reference correction')
t1=time.time()
for i in range(nruns):
    refCor.channelCL(data, 32)
savefile.write('average time to run chanel cor: '+str((time.time()-t1)/float(nruns))+'\n')

print('testing row reference correction')
t1=time.time()
for i in range(nruns):
    refCor.rowCL(data, 4,5)
savefile.write('average time to run row cor: '+str((time.time()-t1)/float(nruns))+'\n')

#test getting saturation info
print('testing get sat counts')
t1=time.time()
for i in range(nruns):
    satCounts = satInfo.getSatCountsCL(data, 0.95, 32)
savefile.write('average time to run getSatCounts: '+str((time.time()-t1)/float(nruns))+'\n')

print('testing get sat frame')
t1=time.time()
for i in range(nruns):
    satFrame = satInfo.getSatFrameCL(data, satCounts,32)
savefile.write('average time to run getSatFrame: '+str((time.time()-t1)/float(nruns))+'\n')

#test up-the-ramp performance
print('testing up the ramp comb')
t1 = time.time()
for i in range(nruns):
    flux = combineData.upTheRampCL(inttime, data, satFrame, 32)
savefile.write('average time to run up the ramp: '+str((time.time()-t1)/float(nruns))+'\n')

#test up-the-ramp performance
print('testing up the ramp CR rejection')
t1 = time.time()
for i in range(nruns):
    flux = combineData.upTheRampCRRejectCL(inttime, data, satFrame, 32)
savefile.write('average time to run up the ramp with CR Rejection: '+str((time.time()-t1)/float(nruns))+'\n')

data = 0

#test other processing stages

flat = wifisIO.readImgFromFile('raytrace_flat.fits')[0]
ronchi = wifisIO.readImgFromFile('raytrace_ronchi.fits')[0]
wave = wifisIO.readImgFromFile('raytrace_arc.fits')[0]

flat = np.random.normal(flat)
ronchi += np.random.normal(ronchi)
wave += np.random.normal(wave)

#testing find limits
print('testing finding limits')
for ncpu in ncpuLst:
    t1=time.time()
    for i in range(nruns):
        limits = slices.findLimits(flat, dispAxis=1, winRng=51, imgSmth=5, limSmth=10,ncpus=ncpu)
    savefile.write('average time to find limits with '+str(ncpu)+' process(es): '+str((time.time()-t1)/float(nruns))+'\n')

wifisIO.writeFits(limits, 'limits.fits')
limits=wifisIO.readImgFromFile('limits.fits')[0]

#extract flat slices
flatSlices = slices.extSlices(flat, limits, dispAxis=1)
wifisIO.writeImgSlices(flat, flatSlices, 'flat_slices.fits')
flatSlices = wifisIO.readImgExtFromFile('flat_slices.fits')[0][1:]

#get dispersion solution
print('testing wavelength mapping')
x0 = 1.31993033e+03
dx = -2.13809325e-01

template = np.loadtxt('wavetemplate.dat')
atlas = 'external_data/best_lines2.dat'
bestLines = np.loadtxt(atlas)

for ncpu in ncpuLst:
    t1 = time.time()
    for i in range(nruns):
        result = waveSol.getWaveSol(wave, template, atlas, 1, [x0,dx], dispAxis=1, winRng=9, mxCcor=30, weights=False, buildSol=True, ncpus=ncpu)
    savefile.write('time to run getWaveSol with '+str(ncpu)+' process(es): '+str((time.time()-t1)/float(nruns))+'\n')

dispSol = []
for sol in result:
    dispSol.append(sol[0])
dispSol = np.array(dispSol)
wifisIO.writeTable(dispSol,'dispSol.dat')
dispSol = wifisIO.readTable('dispSol.dat')

waveMap = waveSol.buildWaveMap(flat, dispSol, dispAxis=1)
waveSlices = slices.extSlices(waveMap, limits, dispAxis=1)
wifisIO.writeImgSlices(waveMap, waveSlices,'wave_slices.fits')
waveSlices = wifisIO.readImgExtFromFile('wave_slices.fits')[0][1:]

#process ronchi image
print('testing ronchi tracing')
ronchiSlices = slices.extSlices(ronchi, limits, dispAxis=1)
wifisIO.writeImgSlices(ronchi, ronchiSlices, 'ronchi_slices.fits')
ronchiSlices=wifisIO.readImgExtFromFile('ronchi_slices.fits')[0][1:]

#benchmark ronchi tracing

for ncpu in ncpuLst:
    t1 = time.time()
    for i in range(nruns):
        ronchiTraces = spatialCor.traceRonchiAll(ronchiSlices, nbin=1, winRng=5, smth=1, bright=False,ncpus=ncpu)
    savefile.write('average time to trace ronchi slices with '+str(ncpu)+' process: '+str((time.time()-t1)/float(nruns))+'\n')

wifisIO.writeImgSlices(ronchi, ronchiTraces, 'ronchi_traces.fits')
ronchiTraces = wifisIO.readImgExtFromFile('ronchi_traces.fits')[0][1:]

#test tracing of zero-point offset
print('testing zero-point wire frame tracing')
zeroImg = wifisIO.readImgFromFile('raytrace_wireframe.fits')[0]
zeroSlices = slices.extSlices(zeroImg, limits, dispAxis=1)

for ncpu in ncpuLst:
    if (ncpu == 1):
        t1 = time.time()
        for i in range(nruns):
            zeroTraces = spatialCor.traceZeroPointAll(zeroSlices, nbin=1, winRng=31, smooth=1, bright=False, MP=False)
        savefile.write('Time to run trace zero point in full serial mode: '+str((time.time()-t1)/float(nruns))+'\n')
    else:
        t1 = time.time()
        for i in range(nruns):
            zeroTraces = spatialCor.traceZeroPointAll(zeroSlices, nbin=1, winRng=31, smooth=1, bright=False, MP=True, ncpus=ncpu)
        savefile.write('Time to run trace zero point in with '+str(ncpu)+' processes: '+str((time.time()-t1)/float(nruns))+'\n')

wifisIO.writeImgSlices(None, zeroTraces, 'zeropoint_traces.fits')
zeroTraces = wifisIO.readImgExtFromFile('zeropoint_traces.fits')[0][1:]

#test trace interpolation/extrapolation
print('testing interpolation/extrapolation for ronchi trace')
for ncpu in ncpuLst:
    t1=time.time()
    for i in range(nruns):
        distSlices = spatialCor.extendTraceAll(ronchiTraces, flatSlices, zeroTraces,space=5., ncpus=ncpu)
    savefile.write('Time to interpolate/extend ronchi trace with '+str(ncpu)+' process: '+str((time.time()-t1)/float(nruns))+'\n')

wifisIO.writeImgSlices(ronchi, distSlices, 'ronchi_fitted.fits')
distSlices = wifisIO.readImgExtFromFile('ronchi_fitted.fits')[0][1:]

#test distortion correction only
print('testing distortion correction')
for ncpu in ncpuLst:
    t1=time.time()
    for i in range(nruns):
        flatCor = createCube.distCorAll(flatSlices, distSlices, ncpus=ncpu)
    savefile.write('average time to distortion correct image with '+str(ncpu)+' process: '+str((time.time()-t1)/float(nruns))+'\n')

wifisIO.writeImgSlices(None, flatCor, 'flatslices_corrected.fits')
flatCor = wifisIO.readImgExtFromFile('flatslices_corrected.fits')[0][1:]

distCor = createCube.distCorAll(distSlices, distSlices)
wifisIO.writeImgSlices(None, distCor, 'distmap_corrected.fits')
distCor = wifisIO.readImgExtFromFile('distmap_corrected.fits')[0][1:]

#test time to get slice trimmed limits
print('testing time to get corrected image trim limits')

for ncpu in ncpuLst:
    t1=time.time()
    for i in range(nruns):
        trimLims = slices.getTrimLimsAll(flatCor,0.75, plot=False, ncpus=ncpu)
    savefile.write('average time to get slice trimmed limits with '+str(ncpu)+' process: '+str((time.time()-t1)/float(nruns))+'\n')

#
##test time to trim slices
print('testing time to trim corrected slices')
for ncpu in ncpuLst:
    if (ncpu == 1):
        t1=time.time()
        for i in range(nruns):
            distTrim = slices.trimSliceAll(distCor, trimLims, MP=False)
        savefile.write('average time to trim slices with serial process:'+str((time.time()-t1)/float(nruns))+'\n')
    else:
        t1=time.time()
        for i in range(nruns):
            distTrim = slices.trimSliceAll(distCor, trimLims, MP=True, ncpus=ncpu)
        savefile.write('average time to trim slices with '+str(ncpu)+' processes: '+str((time.time()-t1)/float(nruns))+'\n')

spatGridProps = createCube.compSpatGrid(distTrim)
np.savetxt('spatGridProps.dat',spatGridProps)
spatGridProps = wifisIO.readTable('spatGridProps.dat')

#test time to trim wavemap
print('testing time to trim wavemap slices')
for ncpu in ncpuLst:
    if (ncpu == 1):
        t1 = time.time()
        for i in range(nruns):
            waveTrim = waveSol.trimWaveSliceAll(waveSlices, flatSlices, 0.1, MP=False)
        savefile.write('average time to trim wavemap slices with serial process: '+str((time.time()-t1)/float(nruns))+'\n')
    else:
        t1 = time.time()
        for i in range(nruns):
            waveTrim = waveSol.trimWaveSliceAll(waveSlices, flatSlices, 0.1, MP=True, ncpus=ncpu)
        savefile.write('average time to trim wavemap slices with '+str(ncpu)+' processes: '+str((time.time()-t1)/float(nruns))+'\n')

waveGridProps = createCube.compWaveGrid(waveTrim,dispSol)
np.savetxt('waveGridProps.dat',waveGridProps)
waveGridProps = wifisIO.readTable('waveGridProps.dat')
wifisIO.writeImgSlices(None, waveTrim, 'wavemap_trim.fits')
waveTrim = wifisIO.readImgExtFromFile('wavemap_trim.fits')[0][1:]

data = wifisIO.readImgFromFile('raytrace_multiplelines2.fits')[0]
dataSlices = slices.extSlices(data, limits, dispAxis=1)

#test time to distortion correct and place on uniform grid
print('testing time to place on uniform spatial and wavelength grid')

for ncpu in ncpuLst:
    t1 = time.time()
    for i in range(nruns):
        dataGrid = createCube.mkWaveSpatGridAll(dataSlices,waveTrim,distSlices,waveGridProps, spatGridProps, ncpus=ncpu)
    savefile.write('average time to grid data with '+str(ncpu)+' process(es): '+str((time.time()-t1)/float(nruns))+'\n')

#test time to create datacube
print('testing time to create final data cube')
for ncpu in ncpuLst:
    if (ncpu == 1):
        t1 = time.time()
        for i in range(nruns):
            dataCube = createCube.mkCube(dataGrid,ndiv=1, MP=False)
        savefile.write('average time to create data cube serialized:'+str((time.time()-t1)/float(nruns))+'\n')
    else:
        t1 = time.time()
        for i in range(nruns):
            dataCube = createCube.mkCube(dataGrid,ndiv=1, MP=True, ncpus=ncpu)
        savefile.write('average time to create data cube with '+str(ncpu)+' processes: '+str((time.time()-t1)/float(nruns))+'\n')

wifisIO.writeFits(dataCube,'data_cube.fits')
wifisIO.writeFits(createCube.collapseCube(dataCube), 'data_cube_collapse.fits')
savefile.write('total time to run all tests: '+str(time.time()-tStart)+'\n')
savefile.close()
