from astropy.io import fits

# Path to the downloaded FITS file
fits_path = r"C:\Users\User\OneDrive - Alpen-Adria Universität Klagenfurt\SS25\623.504 Artificial Intelligence & Machine Learning (SS25)\blackhole-mass-estimation\data\raw\spiders_quasar_bhmass-DR16-v1.fits"

hdul = fits.open(fits_path)

# Display the HDU structure
hdul.info()

# Print the header of the primary HDU
print(hdul[0].header)

# Often the data is in HDU[1]
data = hdul[1].data
print(data.columns)

# Show a sample of the data
print(data[:5])

hdul.close()
