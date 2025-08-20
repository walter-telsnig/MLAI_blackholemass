from astropy.io import fits

fits_path = r"C:\Users\User\OneDrive - Alpen-Adria Universität Klagenfurt\SS25\623.504 Artificial Intelligence & Machine Learning (SS25)\blackhole-mass-estimation\data\raw\spiders_quasar_bhmass-DR16-v1.fits"

with fits.open(fits_path) as hdul:
    data = hdul[1].data
    name_column = data['name']

    # just show first 10 raw bytes
    for i in range(10):
        print(repr(name_column[i]))
