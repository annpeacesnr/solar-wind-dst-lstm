import json
import pickle
import yaml
from pathlib import Path
import tensorflow as tf

from utils.dataset import (
    load_raw, preprocess_features, process_labels, timeseries_dataset_from_df
)
from utils.metrics import rmse

def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    art = cfg["artifacts"]
    out_dir = Path(art["model_dir"])

    # Load artifacts
    model = tf.keras.models.load_model(out_dir / "model")
    with open(out_dir / "scaler.pck", "rb") as f:
        scaler = pickle.load(f)
    effective_cfg = json.loads((out_dir / "config.json").read_text())
    xcols = effective_cfg["features"]["xcols"]
    ycols = effective_cfg["features"]["ycols"]

    # Rebuild preprocessed test set with the same scaler
    solar_wind, dst, sunspots = load_raw(cfg["data"])
    features, _ = preprocess_features(
        solar_wind, sunspots, subset=cfg["features"]["solar_wind_subset"], scaler=scaler
    )
    labels = process_labels(dst, ycols=tuple(ycols))
    data = labels.join(features)

    # Match the split used in train.py
    from utils.dataset import train_val_test_split
    _, _, test = train_val_test_split(data)

    test_ds = timeseries_dataset_from_df(
        test, xcols, ycols, cfg["sequence"]["timesteps"], cfg["sequence"]["batch_size"]
    )

    mse = model.evaluate(test_ds, verbose=0)
    print(f"Test RMSE: {mse**0.5:.2f}")

if __name__ == "__main__":
    main()
