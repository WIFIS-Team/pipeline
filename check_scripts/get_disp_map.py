import wifisWaveSol as waveSol
import wifisIO
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

data = wifisIO.readImgsFromFile('CDSResult.fits')[0]
#whr = np.where(data < 0)
#data[whr] = np.nan

x0 = 1315
dx = -0.213
template = np.loadtxt('template.dat')
atlas = 'external_data/best_lines2.dat'

result = waveSol.getWaveSol(data, template, atlas, 3, [x0,dx], dispAxis=0, winRng=9, mxCcor=30, weights=False, buildSol=False)

#extract results of fitting procedure
dispSol = []
width = []
centFit = []
atlasFit = []
rms = []

for sol in result:
    dispSol.append(sol[0])
    width.append(sol[1])
    centFit.append(sol[2])
    atlasFit.append(sol[3])
    rms.append(sol[4])
    
dispSol = np.asarray(dispSol)
rms = np.asarray(rms)

#build fwhm map
fwhm_map = np.zeros((2048,2048), dtype='float32')
fwhm_map_interp = np.zeros((2048,2048), dtype='float32')

for i in range(2048):
    for j in range(len(centFit[i])):
        fwhm_map[int(centFit[i][j]),i] = width[i][j]

wifisIO.writeFits(fwhm_map, 'fwhm_map.fits')

xgrid = np.arange(2048)
for i in range(2048):
    y = width[i]
    if (len(y) > 0):
        x = centFit[i]
        finter = interp1d(x,y, kind='linear', bounds_error=False)
        ygrid = finter(xgrid)
        fwhm_map_interp[:,i] = ygrid
    
wifisIO.writeFits(fwhm_map_interp, 'fwhm_map_interp.fits')

plt.imshow(fwhm_map_interp, aspect='auto')
plt.colorbar()
plt.savefig('fwhm_map_interp.png', dpi=300)
plt.show()
