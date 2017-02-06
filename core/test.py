import numpy as np
import wifisIO 
import matplotlib.pyplot as plt
import time as time
import os
import wifisRefCor as refCor
import wifisWaveSol as waveSol
from scipy.ndimage.interpolation import shift
import multiprocessing as mp

data = wifisIO.readTable('external_data/wave_test.dat')
atlas = wifisIO.readTable('external_data/ThAr_mask.dat')
best = wifisIO.readTable('external_data/strong_lines2.dat')

#exclude lines that are outside of detector
whr = np.where(best[:,1] >= 0)
best = best[whr[0],:]
whr = np.where(best[:,1] < 2048)
best = best[whr[0],:]

yRow = data[:,1]

#fit = getWaveSol.getSolFullRow([yRow, best, atlas])

ncpus =mp.cpu_count()

pool = mp.Pool(ncpus)

lst = []

for i in range(100):
    ytmp = yRow+np.random.normal(0, 10.0, size=len(yRow))
    yinp = shift(ytmp, np.random.normal(0, 10.0))
    lst.append([yinp,best, atlas, yRow,1])

t1 = time.time()
result = pool.map(waveSol.getSolQuickRow, lst)

pool.close()

print(time.time() - t1)

parms = np.array(result)

p1 = np.median(parms[:,0])
p2 = np.median(parms[:,1])

#print('median of p0',p1, 'with std', np.std(parms[:,0]))
print('median of p1', p2, 'with std' , np.std(parms[:,1]))

for i in range(parms.shape[1]):
    plt.plot(parms[:,i])
    plt.show()

    
