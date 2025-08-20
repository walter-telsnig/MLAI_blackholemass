from astropy.io import fits

fits_path = r"C:\Users\User\OneDrive - Alpen-Adria Universität Klagenfurt\SS25\623.504 Artificial Intelligence & Machine Learning (SS25)\blackhole-mass-estimation\data\raw\spiders_quasar_bhmass-DR16-v1.fits"

with fits.open(fits_path) as hdul:
    data = hdul[1].data
    name_col = data["name"]

    for i in range(5):
        print(f"RAW VALUE: {repr(name_col[i])}")
        print(f"AS STRING: {str(name_col[i])}")
