from astropy.io import fits

# correct path to your downloaded file
fits_path = r"C:\Users\User\OneDrive - Alpen-Adria Universität Klagenfurt\SS25\623.504 Artificial Intelligence & Machine Learning (SS25)\blackhole-mass-estimation\data\raw\spec-2829-54523-0376.fits"

# open the FITS file
hdul = fits.open(fits_path)

# show its structure
hdul.info()

# print the first HDU header
print(hdul[0].header)

# typically the data table is in HDU[1]
data = hdul[1].data
print(data.columns)

# show the first few rows
print(data[:5])

# close the file
hdul.close()
