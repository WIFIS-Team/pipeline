"""

Tasks to modify FITS headers

"""

from astropy import wcs
from astropy.io import fits
import colorama
from astropy import coordinates as coord, units
import numpy as np

def getWCSCube(data, hdr, xScale, yScale, waveGridProps, useSesameCoords=False):
    """
    Returns the corresponding WCS header parameters based on given input parameters.
    Usage: header = getWCSCube(data, hdr, xScale, yScale, waveGridProps, useSesameCoords=False)
    data is the input data cube
    hdr is the input astropy header object for the data cube
    xScale is the pixel scale in arcsec along the slice direction.
    yScale is the pixel scale in arcsec perpendicular to the slice direction
    waveGridProps is a list containing the starting wavelength, ending wavelength and number of pixels along the dispersion direction.
    useSesameCoords is a boolean flag to specify if the centre coordinat should be replaced by a lookup value corresponding to the object name's Sesame coordinates instead of the telescope pointing coordinates. If object name cannot be found, the fallback is to use the telescope pointing coordinates.
    """

    #compute the properties of the wavelength grid based on the waveGridProps provided
    dWave = (waveGridProps[1]-waveGridProps[0])/(waveGridProps[2]-1)

    #initiate WCS object
    w = wcs.WCS(naxis=3)

    if useSesameCoords:
        #use astropy SESAME name resolver lookup functionality to set the central RA/DEC instead of using the telescope position
        try:
            objName = hdr['OBJECT']
            coords = coord.SkyCoord.from_name(objName)
            telRA = coords.ra.deg
            telDEC = coords.dec.deg
        #if any errors pop up, continue using the telescope coordinates instead
        except:
            useSesameCoords=False

    if not useSesameCoords:
        #convert input strings to degrees
        telRA = hdr['RA_DEG']
        telDEC = hdr['DEC_DEG']

    #get specific details to add to the header
    rotAngle = hdr['CRPA']

    #rotAngle of 90 corresponds to N-S alignment
    #rotAngle of 180 corresponds to W-E alignment
    
    w.wcs.cdelt = [xScale/3600., yScale/3600., dWave]

    #central pixel along y-axis is always first slice
    #w.wcs.crpix = [data.shape[1]/2., data.shape[0]/2., 1]
    centSlice = (data.shape[0]-1.)/17.*8.
    #w.wcs.crpix = [data.shape[1]/2., data.shape[0]/2., 1]
    centSlice = [data.shape[1]/2.,centSlice, 1]
    w.wcs.crval=[telRA,telDEC, waveGridProps[0]]
    w.wcs.crota=[float(rotAngle), float(rotAngle),0.]
    w.wcs.ctype=["RA---TAN","DEC--TAN","WAVE"]
    w.wcs.cunit=["deg","deg","nm"]
    header = w.to_header()

    #update header or add if keyword doesn't exist
    for h in header.cards:
        hdr.set(h[0],h[1],h[2])

    #Now make sure NAXIS keywords are correct
    #hdr.set('NAXIS',3)
    #hdr.set('NAXIS1',data.shape[1])
    #hdr.set('NAXIS2',data.shape[0])
    #if not 'NAXIS3' in hdr:
    #    hdr.insert('NAXIS2', ('NAXIS3', int(waveGridProps[2])))
    #else:
    #    hdr.set('NAXIS3',data.shape[2])
        
    return

def getWCSImg(data, hdr, xScale, yScale, useSesameCoords=False):
    """
    Returns the corresponding WCS header parameters based on given input parameters for the given image
    Usage: header = getWCSImg(data, hdr, xScale, yScale, useSesameCoords=False)
    data is the input data image
    hdr is the input astropy header object for the data cube
    xScale is the pixel scale in arcsec along the slice direction.
    yScale is the pixel scale in arcsec perpendicular to the slice direction
    useSesameCoords is a boolean flag to specify if the centre coordinat should be replaced by a lookup value corresponding to the object name's Sesame coordinates instead of the telescope pointing coordinates. If object name cannot be found, the fallback is to use the telescope pointing coordinates.
    """ 

    #initiate WCS object
    w = wcs.WCS(naxis=2)

    if useSesameCoords:
        #use astropy SESAME name resolver lookup functionality to set the central RA/DEC instead of using the telescope position
        try:
            objName = hdr['OBJECT']
            coords = coord.SkyCoord.from_name(objName)
            telRA = coords.ra.deg
            telDEC = coords.dec.deg
        except:
            useSesameCoords=False

    if not useSesameCoords:
        #convert input strings to degrees
        telRA = hdr['RA_DEG']
        telDEC = hdr['DEC_DEG']

    rotAngle = hdr['CRPA']

    #rotAngle of 90 corresponds to N-S alignment
    #rotAngle of 180 corresponds to W-E alignment 
    #convert input strings to degrees
    
    w.wcs.cdelt = [xScale/3600., yScale/3600.]
    centSlice = (data.shape[0]-1.)/17.*8.
    #w.wcs.crpix = [data.shape[1]/2., data.shape[0]/2.]
    w.wcs.crpix = [data.shape[1]/2., centSlice]
    w.wcs.crval=[telRA,telDEC]
    w.wcs.crota=[float(rotAngle), float(rotAngle)]
    w.wcs.ctype=["RA---TAN","DEC--TAN"]
    w.wcs.cunit=["deg","deg"]
    header = w.to_header()

    #update header or add if keyword doesn't exist
    for h in header.cards:
        hdr.set(h[0],h[1],h[2])

    #update NAXIS parameters
    #hdr['NAXIS1'] = data.shape[1]
    #hdr['NAXIS2'] = data.shape[0]
    return

def addTelInfo(hdr, obsinfoFile, logfile=None, obsCoords=None):
    """
    Routine to add telescope info from obsinfo.dat file to astropy header object
    Usage: addTelInfo(hdr, obsinfoFile, logfile=None, obsCoords=None)
    hdr is an astropy header object
    obsinfoFile is the location to the obsinfo.dat file to read
    logfile is an optional file object for writing to a logfile
    obsCoords is an optional list containing the observatory coordinates [longitude (deg), latitude (deg), altitude (m)]
    """

    colorama.init()

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
    hdr.set('EPOCH', 2000, 'Target coordinate system')
    hdr.set('EQUINOX', 2000, 'Equinox of coordinate system')

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

    #make sure that rotator angle uses reasonable values, if not assume the angle is 90 deg
    try:
        crpa = float(dct['IIS'])
    except ValueError:
        print(colorama.Fore.RED+'*** WARNING: ROTATOR POSITION ANGLE FORMAT INCORRECT, ASSUMING 90 DEG ***'+colorama.Style.RESET_ALL)
        if logfile is not None:
            logfile.write('*** WARNING: ROTATOR POSITION ANGLE FORMAT INCORRECT, ASSUMING 90 DEG ***\n')
        crpa = 90.
    
    hdr.set('CRPA', crpa, 'Cass Rotator Position Angle at end')
    hdr.set('HA', dct['HA'], 'Telescope hour angle')
    hdr.set('ST', dct['ST'], 'Sidereal time at end of observation')
    hdr.set('UT', dct['UT'],'Universal time at end of observation')
    hdr.set('JD', float(dct['JD']), 'Julian date at end of observation')
    hdr.set('LT', dct['Timestamp'], 'Local time stamp at end of observation')

    if obsCoords is not None:
        #compute paralactic angle using definite of EQ 10 of Filippenko 1982, PASP 94,715
        #print('Determining paralactic angle')
        #logfile.write('Determining paralactic angle\n')
        haAng = coord.Angle(hdr['HA']+' hour')
        decAng = coord.Angle(hdr['DEC']+' degrees')
        latAng = coord.Angle(str(obsCoords[1])+' degrees')
        sin_eta = (np.sin(haAng.rad)*np.cos(latAng.rad))/np.sqrt(1. - (np.sin(latAng.rad)*np.sin(decAng.rad) + np.cos(latAng.rad)*np.cos(decAng.rad)*np.cos(haAng.rad))**2)
        eta = coord.Angle(str(np.arcsin(sin_eta)) + ' rad')
        hdr.set('PA_ANG',eta.deg, 'Paralactic angle, in degrees')
     
    #finally, add the obsinfo.dat file information directly
    for line in linesSplit:
        out = ''
        for word in line:
            out += word+' '
        hdr.add_comment(out)

    return
    
