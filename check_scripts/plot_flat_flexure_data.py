import matplotlib
matplotlib.use('gtkagg')
import wifisIO
import numpy as np
import matplotlib.pyplot as plt
import os

#****************************************************************************************
#REQUIRED INPUT FILES
flatLstFile = 'flat.lst'

#****************************************************************************************

if os.path.exists(flatLstFile):
    flatLst = wifisIO.readAsciiList(flatLstFile)

    if flatLst.ndim == 0:
        flatLst = np.asarray([flatLst])

else:
    raise SystemExit('*** FAILURE: NO FLAT LIST INPUT FILE FOUND ***')

shft = []
pa = []

for flat in flatLst:
    hdr = wifisIO.readImgsFromFile('processed/'+flat+'_flat_limits.fits')[1]

    shft.append(hdr['LIMSHIFT'])
    pa.append(hdr['PA_ANG'])

    
plt.plot(pa, shft, 'o')
plt.title('Slice limits shift versus paralactic angle')
plt.ylabel('Shift [pixels]')
plt.xlabel('Paralactic angle [deg]')
plt.savefig('flexure_limits_vs_pa.png')
plt.show()

    
