import pandas as pd

csv_in = r"C:\Users\User\OneDrive - Alpen-Adria Universität Klagenfurt\SS25\623.504 Artificial Intelligence & Machine Learning (SS25)\blackhole-mass-estimation\data\processed\spiders_quasar_sample.csv"

# bulletproof read with latin1
df = pd.read_csv(csv_in, encoding="latin1", on_bad_lines="skip")

# Select only columns of interest - these are the features we will use for training first model (randomforest)
selected_cols_old = [
    "name",
    "redshift",
    "edd_ratio1",
    "l5100",
    "fwhm1_hb",
    "logBHMA_hb"
]

# Extensions to features for the XGBoost model
selected_cols = [
    "name",
    "redshift",
    "edd_ratio1",
    "l5100",
    "l3000",
    "fwhm1_hb",
    "virialfwhm_mgII",
    "logBHMA_hb"
]

df = df[selected_cols]

# Filter out missing or placeholder values
df = df[
    (df["logBHMA_hb"] > 0) &
    (df["l5100"] > 0) &
    (df["fwhm1_hb"] > 0)
]

# Preview
print(df.head())

csv_out = r"C:\Users\User\OneDrive - Alpen-Adria Universität Klagenfurt\SS25\623.504 Artificial Intelligence & Machine Learning (SS25)\blackhole-mass-estimation\data\processed\spiders_quasar_features.csv"
df.to_csv(csv_out, index=False)

print(f"✅ Features CSV saved to {csv_out}")
