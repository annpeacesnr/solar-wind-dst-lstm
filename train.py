import json
import pickle
import yaml
from pathlib import Path

from utils.dataset import (
    load_raw, preprocess_features, process_labels, build_feature_columns,
    train_val_test_split, timeseries_dataset_from_df
)
from models.lstm_model import build_lstm
from utils.plot import plot_history

def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    # --- Load raw data
    solar_wind, dst, sunspots = load_raw(cfg["data"])

    # --- Preprocess
    sw_subset = cfg["features"]["solar_wind_subset"]
    features, scaler = preprocess_features(
        solar_wind, sunspots, subset=sw_subset, stats=tuple(cfg["features"]["agg_stats"])
    )
    xcols = build_feature_columns(sw_subset)
    ycols = cfg["features"]["ycols"]
    labels = process_labels(dst, ycols=tuple(ycols))
    data = labels.join(features)
    assert (data[xcols].isna().sum() == 0).all()

    train, val, test = train_val_test_split(data)

    # --- Datasets
    timesteps = cfg["sequence"]["timesteps"]
    batch_size = cfg["sequence"]["batch_size"]
    train_ds = timeseries_dataset_from_df(train, xcols, ycols, timesteps, batch_size)
    val_ds   = timeseries_dataset_from_df(val,   xcols, ycols, timesteps, batch_size)

    # --- Model
    model_cfg = cfg["model"]
    model = build_lstm(
        input_timesteps=timesteps,
        input_features=len(xcols),
        output_dim=len(ycols),
        neurons=model_cfg["neurons"],
        dropout=model_cfg["dropout"],
        stateful=model_cfg["stateful"],
    )
    model.compile(loss=model_cfg["loss"], optimizer=model_cfg["optimizer"])
    model.summary()

    # --- Train
    history = model.fit(
        train_ds,
        epochs=model_cfg["epochs"],
        validation_data=val_ds,
        verbose=1,
        shuffle=False,
    )

    # --- Save artifacts
    out_dir = Path(cfg["artifacts"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model")
    with open(out_dir / "scaler.pck", "wb") as f:
        pickle.dump(scaler, f)
    # persist effective runtime config
    effective_cfg = cfg.copy()
    effective_cfg["features"]["xcols"] = xcols
    (out_dir / "config.json").write_text(json.dumps(effective_cfg, indent=2))

    # --- Plot
    plot_history(history)

if __name__ == "__main__":
    main()
