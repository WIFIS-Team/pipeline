"""

Tasks to modify FITS headers

"""

from astropy import wcs


def getWCSCube(data, telRA, telDEC, xScale, yScale, rotAngle, waveGridProps):
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
    RA = float(telRA[0:2])*15. + float(telRA[2:4])*15/60. + float(telRA[4:])*15/3600.
    DEC = float(telDEC[0:3]) + float(telDEC[3:5])/60. + float(telDEC[5:])/3600.

    #rotAngle of 90 corresponds to N-S alignment
    #rotAngle of 180 corresponds to W-E alignment
    
    w.wcs.cdelt = [xScale/3600., -yScale/3600., dWave]
    w.wcs.crpix = [data.shape[1]/2., data.shape[0]/2., 1]
    w.wcs.crval=[telRA,telDEC, waveGridProps[0]]
    w.wcs.crota=[float(rotAngle), float(rotAngle),0.]
    w.wcs.ctype=["RA---AZP","DEC--AZP","WAVE"]
    w.wcs.cunit=["deg","deg","nm"]
    header = w.to_header()

    return header

def getWCSImg(data, telRA, telDEC, xScale, yScale, rotAngle):
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

    #convert input strings to degrees
    RA = float(telRA[0:2])*15. + float(telRA[2:4])*15/60. + float(telRA[4:])*15/3600.
    DEC = float(telDEC[0:3]) + float(telDEC[3:5])/60. + float(telDEC[5:])/3600.

    #rotAngle of 90 corresponds to N-S alignment
    #rotAngle of 180 corresponds to W-E alignment
    
    w.wcs.cdelt = [xScale/3600., -yScale/3600.]
    w.wcs.crpix = [data.shape[1]/2., data.shape[0]/2.]
    w.wcs.crval=[telRA,telDEC]
    w.wcs.crota=[float(rotAngle), float(rotAngle)]
    w.wcs.ctype=["RA---TAN","DEC--TAN"]
    w.wcs.cunit=["deg","deg"]
    header = w.to_header()
        
    return header

