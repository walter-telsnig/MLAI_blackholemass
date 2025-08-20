from astropy.io import fits
import pandas as pd
import numpy as np

fits_path = r"C:\Users\User\OneDrive - Alpen-Adria Universität Klagenfurt\SS25\623.504 Artificial Intelligence & Machine Learning (SS25)\blackhole-mass-estimation\data\raw\spiders_quasar_bhmass-DR16-v1.fits"

with fits.open(fits_path) as hdul:
    data = hdul[1].data

    # isolate name as list of strings from astropy
    name_column = [str(x).strip() for x in data["name"]]

    # build numeric columns
    table = {}
    for name in data.names:
        if name != "name":
            column = data[name].byteswap().view(data[name].dtype.newbyteorder('='))
            table[name] = column

    # build DataFrame from numeric columns
    df = pd.DataFrame(table)

    # inject correct name
    df.insert(0, "name", name_column)

    preview_cols = [
        "name", "redshift", "logBHMA_hb", "logBHMS_mgII", "edd_ratio1"
    ]
    print(df[preview_cols].head())

    csv_out = r"C:\Users\User\OneDrive - Alpen-Adria Universität Klagenfurt\SS25\623.504 Artificial Intelligence & Machine Learning (SS25)\blackhole-mass-estimation\data\processed\spiders_quasar_sample.csv"
    df.to_csv(csv_out, index=False)

    print(f"✅ CSV saved to {csv_out}")
