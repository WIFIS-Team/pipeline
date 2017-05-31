"""

Tasks to modify FITS headers

"""

from astropy import wcs
from astropy.io import fits

def getWCSCube(data, hdr, xScale, yScale, waveGridProps):
    """
    Returns the corresponding WCS header parameters based on given input parameters.
    Usage: header = getWCSCube(data, telRA, telDEC, RAScale, DECScale, waveGridProps)
    data is the input data cube
    telRA is a string containing the telescope pointing RA coordinate as 'hhmmss.ss'
    telDEC is a string containing the telescope pointing DEC coordinate as '+ddmmss.ss'
    xScale is the pixel scale in arcsec along the slice direction.
    yScale is the pixel scale in arcsec perpendicular to the slice direction
    rotAngle is the rotation angle ... in ...
    waveGridProps is a list containing the starting wavelength, ending wavelength and number of pixels along the dispersion direction.
    """

    dWave = (waveGridProps[1]-waveGridProps[0])/(waveGridProps[2]-1)
    w = wcs.WCS(naxis=3)
    
    #convert input strings to degrees
    telRA = hdr['RA_DEG']
    telDEC = hdr['DEC_DEG']
    rotAngle = hdr['CRPA']

    #rotAngle of 90 corresponds to N-S alignment
    #rotAngle of 180 corresponds to W-E alignment
    
    w.wcs.cdelt = [xScale/3600., -yScale/3600., dWave]
    w.wcs.crpix = [data.shape[1]/2., data.shape[0]/2., 1]
    w.wcs.crval=[telRA,telDEC, waveGridProps[0]]
    w.wcs.crota=[float(rotAngle), float(rotAngle),0.]
    w.wcs.ctype=["RA---TAN","DEC--TAN","WAVE"]
    w.wcs.cunit=["deg","deg","nm"]
    header = w.to_header()

    #update header or add if keyword doesn't exist
    for h in header.cards:
        hdr.set(h[0],h[1],h[2])
    
    return

def getWCSImg(data, hdr, xScale, yScale):
    """
    Returns the corresponding WCS header parameters based on given input parameters.
    Usage: header = getWCSImg(data, telRA, telDEC, RAScale, DECScale)
    data is the input data image
    telRA is a string containing the telescope pointing RA coordinate as 'hhmmss.ss'
    telDEC is a string containing the telescope pointing DEC coordinate as '+ddmmss.ss'
    RAscale is the RA pixel scale in arcsec
    DECscale is the DEC pixel scale in arcsec
    rotAngle is the rotation angle ... in ...
    """

    w = wcs.WCS(naxis=2)

    #get telescope info
    telRA = hdr['RA_DEG']
    telDEC = hdr['DEC_DEG']
    rotAngle = hdr['CRPA']

    #rotAngle of 90 corresponds to N-S alignment
    #rotAngle of 180 corresponds to W-E alignment 
    #convert input strings to degrees
    
    w.wcs.cdelt = [xScale/3600., -yScale/3600.]
    w.wcs.crpix = [data.shape[1]/2., data.shape[0]/2.]
    w.wcs.crval=[telRA,telDEC]
    w.wcs.crota=[float(rotAngle), float(rotAngle)]
    w.wcs.ctype=["RA---TAN","DEC--TAN"]
    w.wcs.cunit=["deg","deg"]
    header = w.to_header()

    #update header or add if keyword doesn't exist
    for h in header.cards:
        hdr.set(h[0],h[1],h[2])
    
    return

def addTelInfo(hdr, obsinfoFile):
    """
    """

    #open the obsinfo.dat file associated with the data
    fle = open(obsinfoFile,'r')

    #split the data into more useable lines
    lines = fle.readlines()
    linesSplit = []

    #now go through each line and extract the header keys and their values
    lst = []
    for line in lines:
        
        linesSplit.append(line.split())
       
        word = linesSplit[-1][0]
        pos = word.find(':')
    
        if (pos > 0):
            tmpKey = word[0:pos]
            i=1
        else:
            tmpKey = word+' '
            i=1
            while (i < len(linesSplit[-1])):
                word = linesSplit[-1][i]
                pos = word.find(':')
                if (pos > 0):
                    tmpKey += word[0:pos]
                    i+=1
                    break
                else:
                    tmpKey += word + ' '
                    i+=1
                
        tmpValue = ''
        for j in range(i, len(linesSplit[-1])):
            tmpValue += linesSplit[-1][j] + ' '
   
        lst.append([tmpKey, tmpValue])
    
    fle.close()

    #create dictionary from list for easier use
    dct = dict(lst)

    #add the following keywords to FITS header
    hdr.set('INSTRUME','WIFIS','Instrument name')
    hdr.set('OBJECT', dct['Source'], 'Object name')
    hdr.set('OBSCLASS', dct['Obs Type'], 'Observe class')
    hdr.set('OBSERVAT', dct['ID'], 'Name of telescope')
    hdr.set('EPOCH', 2000., 'Target coordinate system')
    hdr.set('EQUINOX', 2000., 'Equinox of coordinate system')

    #convert target RA and DEC into degrees
    RA_deg = float(dct['RA'][0:2])*15. + float(dct['RA'][2:4])*15/60. + float(dct['RA'][4:])*15/3600.
    DEC_deg = float(dct['DEC'][0:3]) + float(dct['DEC'][3:5])/60. + float(dct['DEC'][5:])/3600.

    RA = dct['RA'][0:2]+':'+ dct['RA'][2:4] + ':'+ dct['RA'][4:]
    DEC = dct['DEC'][0:3] + ':'+ dct['DEC'][3:5] + ':'+ dct['DEC'][5:]
    hdr.set('RA', RA, 'Telescope right ascension')
    hdr.set('DEC', DEC, 'Telescope declination')
    hdr.set('RA_DEG', RA_deg, 'Telescope right ascension in degrees')
    hdr.set('DEC_DEG', DEC_deg, 'Telescope declination in degrees')
    hdr.set('ELEVATIO', float(dct['EL']), 'Current Elevation')
    hdr.set('AZIMUTH', float(dct['AZ']), 'Current Azimuth')

    try:
        crpa = float(dct['IIS'])
    except ValueError:
        print('***WARNING. ROTATOR POSITION ANGLE FORMAT INCORRECT, ASSUMING 90 DEG')
        crpa = 90.
    
    hdr.set('CRPA', crpa, 'Cass Rotator Position Angle at end')
    hdr.set('HA', dct['HA'], 'Telescope hour angle')
    hdr.set('ST', dct['ST'], 'Sidereal time at end of observation')
    hdr.set('UT', dct['UT'],'Universal time at end of observation')
    hdr.set('JD', dct['JD'], 'Julian date at end of observation')
    hdr.set('LT', dct['Timestamp'], 'Local time stamp at end of observation')
    
    #finally, add the obsinfo.dat file information directly
    for line in linesSplit:
        out = ''
        for word in line:
            out += word+' '
        hdr.add_comment(out)

    return
    
