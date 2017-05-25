"""
"""


from astropy import wcs



def getWCSCube(data, telRA, telDEC, RAscale, DECscale, waveGridProps):
    """
    """

    dWave = (waveGridProps[1]-waveGridProps[0])/(waveGridProps[2]-1)
    
    w = wcs.WCS(naxis=3)
    #w.wcs.cdelt = [(50./float(data.shape[1]))/3600., (20./float(data.shape[0]))/3600.,waveGridProps[2]]
    w.wcs.cdelt = [RAscale, DECscale, dWave]
    w.wcs.crpix = [data.shape[1]/2., data.shape[0]/2., 1]
    w.wcs.crval=[telRA,telDEC, waveGridProps[0]]
    w.wcs.ctype=["RA---TAN","DEC--TAN","WAVE"]
    w.wcs.cunit=["deg","deg","nm"]
    header = w.to_header()

    return header

def getWCSImg(data, telRA, telDEC, RAscale, DECscale):
    """
    """

    w = wcs.WCS(naxis=3)
    #w.wcs.cdelt = [(50./float(data.shape[1]))/3600., (20./float(data.shape[0]))/3600.,waveGridProps[2]]
    w.wcs.cdelt = [RAscale, DECscale]
    w.wcs.crpix = [data.shape[1]/2., data.shape[0]/2., 1]
    w.wcs.crval=[telRA,telDEC, waveGridProps[0]]
    w.wcs.ctype=["RA---TAN","DEC--TAN"]
    w.wcs.cunit=["deg","deg"]
    header = w.to_header()

    return header

