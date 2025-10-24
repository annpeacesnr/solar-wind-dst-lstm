import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def _read_indexed_csv(path):
    df = pd.read_csv(path)
    df["timedelta"] = pd.to_timedelta(df["timedelta"])
    df.set_index(["period", "timedelta"], inplace=True)
    return df

def load_raw(data_cfg):
    base = data_cfg["raw_dir"]
    solar_wind = _read_indexed_csv(os.path.join(base, data_cfg["solar_wind"]))
    dst = _read_indexed_csv(os.path.join(base, data_cfg["dst"]))
    sunspots = _read_indexed_csv(os.path.join(base, data_cfg["sunspots"]))
    return solar_wind, dst, sunspots

def impute_features(feature_df):
    # forward-fill monthly sunspot values, interpolate others
    feature_df["smoothed_ssn"] = feature_df["smoothed_ssn"].fillna(method="ffill")
    return feature_df.interpolate()

def aggregate_hourly(feature_df, stats=("mean", "std")):
    agged = feature_df.groupby(
        ["period", feature_df.index.get_level_values(1).floor("H")]
    ).agg(list(stats))
    agged.columns = ["_".join(x) for x in agged.columns]
    return agged

def preprocess_features(solar_wind, sunspots, subset, stats=("mean","std"), scaler=None):
    if subset:
        solar_wind = solar_wind[subset]
    hourly_features = aggregate_hourly(solar_wind, stats).join(sunspots)
    if scaler is None:
        scaler = StandardScaler().fit(hourly_features)
    norm = pd.DataFrame(
        scaler.transform(hourly_features),
        index=hourly_features.index,
        columns=hourly_features.columns,
    )
    imputed = impute_features(norm)
    return imputed, scaler

def build_feature_columns(solar_wind_subset):
    xcols = [c + "_mean" for c in solar_wind_subset] + [c + "_std" for c in solar_wind_subset] + ["smoothed_ssn"]
    return xcols

def process_labels(dst, ycols=("t0","t1")):
    y = dst.copy()
    y["t1"] = y.groupby("period")["dst"].shift(-1)
    y.columns = list(ycols)
    return y

def train_val_test_split(df, test_per_period=6000, val_per_period=3000):
    test = df.groupby("period").tail(test_per_period)
    interim = df[~df.index.isin(test.index)]
    val = df.groupby("period").tail(val_per_period)
    train = interim[~interim.index.isin(val.index)]
    return train, val, test

def timeseries_dataset_from_df(df, xcols, ycols, timesteps, batch_size):
    dataset = None
    for _, period_df in df.groupby("period"):
        inputs = period_df[xcols][:-timesteps]
        outputs = period_df[ycols][timesteps:]
        period_ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            inputs, outputs, timesteps, batch_size=batch_size
        )
        dataset = period_ds if dataset is None else dataset.concatenate(period_ds)
    return dataset
