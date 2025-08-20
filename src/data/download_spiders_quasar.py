import os
import urllib.request

onecloud_data_dir = r"C:\Users\User\OneDrive - Alpen-Adria Universität Klagenfurt\SS25\623.504 Artificial Intelligence & Machine Learning (SS25)\blackhole-mass-estimation\data\raw"

os.makedirs(onecloud_data_dir, exist_ok=True)

# DR16 SPIDERS quasar black hole mass catalog
url = "https://data.sdss.org/sas/dr16/eboss/spiders/analysis/spiders_quasar_bhmass-DR16-v1.fits"

local_file = os.path.join(onecloud_data_dir, "spiders_quasar_bhmass-DR16-v1.fits")

try:
    print(f"⬇ Downloading {url} ...")
    urllib.request.urlretrieve(url, local_file)
    print(f"✅ Download complete: {local_file}")
except Exception as e:
    print(f"❌ Download failed: {e}")
