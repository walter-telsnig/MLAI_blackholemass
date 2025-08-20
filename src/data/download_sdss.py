import os
import urllib.request

# OneCloud raw data directory
onecloud_data_dir = r"C:\Users\User\OneDrive - Alpen-Adria Universität Klagenfurt\SS25\623.504 Artificial Intelligence & Machine Learning (SS25)\blackhole-mass-estimation\data\raw"

# make sure the directory exists
os.makedirs(onecloud_data_dir, exist_ok=True)

# DR17 confirmed spectrum
#sdss_url = "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/2829/spec-2829-54523-0376.fits"
sdss_url = "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/2829/spec-2829-54623-0001.fits"

# local file path
local_file_path = os.path.join(onecloud_data_dir, "spec-2829-54523-0376.fits")

try:
    print(f"⬇ Downloading {sdss_url} ...")
    urllib.request.urlretrieve(sdss_url, local_file_path)
    print(f"✅ Download complete: {local_file_path}")
except Exception as e:
    print(f"❌ Download failed: {e}")
