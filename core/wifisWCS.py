"""
"""


from astropy import wcs



def getWCS(data, telRA, telDEC, RAscale, DECscale, waveGridProps):
    """
    """

    w = wcs.WCS(naxis=3)
    #w.wcs.cdelt = [(50./float(data.shape[1]))/3600., (20./float(data.shape[0]))/3600.,waveGridProps[2]]
    w.wcs.cdelt = [RAcale, DECcale, waveGridProps[2]]
    w.wcs.crpix = [data.shape[1]/2., data.shape[0]/2., 0]
    w.wcs.crval=[telRA,telDEC, waveGridProps[0]]
    w.wcs.ctype=["RA---TAN","DEC--TAN","WAVE"]
    header = w.to_header()

    return header
