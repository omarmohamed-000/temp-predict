# 🌡️ Urban Heat Island Temperature Predictor

An end-to-end machine learning project that predicts **urban surface temperature** from tabular climate and geospatial features, with an interactive **Gradio** web app for making predictions.

- **Model:** XGBoost regressor inside a scikit-learn `Pipeline` (with `StandardScaler`)
- **Evaluation:** K-fold cross-validation + hold-out test set (RMSE and R²)
- **Interpretability:** gain-based and permutation feature importances
- **App:** Gradio UI with synced sliders and number boxes, plus a clear prediction card

---

## Project structure

```
.
├── urban_temp_model.py     # Training script (preprocessing, CV, evaluation, saving)
├── app.py                  # Gradio web application
├── requirements.txt        # Python dependencies
├── README.md
└── models/                 # Created after training
    ├── urban_temp_model.joblib
    └── metrics.json
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the model

With your own data:

```bash
python urban_temp_model.py \
  --data data/urban_data.csv \
  --target surface_temp_c
```

Without data (uses a built-in synthetic dataset, good for a quick test):

```bash
python urban_temp_model.py
```

### Training options

| Argument       | Default             | Description                              |
|----------------|---------------------|------------------------------------------|
| `--data`       | *(synthetic data)*  | Path to a CSV or Parquet file            |
| `--target`     | `surface_temp_c`    | Name of the target column                |
| `--features`   | built-in list       | Space-separated feature column names     |
| `--test-size`  | `0.2`               | Hold-out fraction                        |
| `--cv-folds`   | `5`                 | Number of cross-validation folds         |
| `--out-dir`    | `models`            | Where to save the model and metrics      |

### Expected data format

One row per location/observation. Default feature columns:

| Column               | Description                          | Typical range |
|----------------------|--------------------------------------|---------------|
| `green_space_ratio`  | Vegetation coverage                  | 0 – 1         |
| `building_density`   | Built-up coverage                    | 0 – 1         |
| `wind_speed`         | Mean wind speed (m/s)                | 0 – 20        |
| `albedo`             | Surface reflectivity                 | 0 – 1         |
| `air_temperature`    | Ambient air temperature (°C)         | –10 – 50      |
| `humidity`           | Relative humidity (%)                | 0 – 100       |
| `elevation`          | Elevation (m)                        | 0 – 1000      |
| `distance_to_water`  | Distance to nearest water body (m)   | 0 – 10000     |
| `sky_view_factor`    | Visible sky fraction                 | 0 – 1         |
| `surface_temp_c`     | **Target** – surface temperature (°C)| –             |

Using different features? Pass them with `--features` and the app will adapt automatically, since it reads the feature list stored in the saved model. Features not in the app's built-in spec get a generic 0–100 slider; edit `FEATURE_SPECS` in `app.py` to set proper ranges.

### What training does

1. Validates columns, drops rows with a missing target (the label is never imputed)
2. Splits into train/test sets
3. Runs K-fold CV on the training set only
4. Fits the final pipeline and evaluates on the hold-out set (RMSE, R²)
5. Prints gain and permutation feature importances
6. Saves the pipeline, feature list, target name and metrics with `joblib`

## 2. Launch the web app

```bash
python app.py
```

Then open <http://127.0.0.1:7860>.

| Argument   | Default                             | Description                    |
|------------|-------------------------------------|--------------------------------|
| `--model`  | `models/urban_temp_model.joblib`    | Path to the saved model bundle |
| `--host`   | `127.0.0.1`                         | Use `0.0.0.0` to expose on LAN |
| `--port`   | `7860`                              | Server port                    |
| `--share`  | off                                 | Create a temporary public link |

### Using the app

1. Set each feature with its slider, or type an exact value in the box beside it (they stay in sync)
2. Click **Predict**
3. Read the predicted temperature, its heat category (Mild / Warm / Hot / Extreme heat) and the model's typical error
4. Click **Reset** to restore default values

## Using the model in your own code

```python
import joblib
import pandas as pd

bundle = joblib.load("models/urban_temp_model.joblib")
new_data = pd.DataFrame([{
    "green_space_ratio": 0.2, "building_density": 0.7, "wind_speed": 2.0,
    "albedo": 0.15, "air_temperature": 33, "humidity": 50,
    "elevation": 80, "distance_to_water": 2000, "sky_view_factor": 0.5,
}])
print(bundle["pipeline"].predict(new_data[bundle["features"]]))
```

## Deployment

- **Hugging Face Spaces:** create a Gradio Space, upload `app.py`, `requirements.txt` and `models/urban_temp_model.joblib`. Rename `app.py` if needed and make sure the model path matches.
- **Docker / server:** run `python app.py --host 0.0.0.0 --port 7860` and put it behind a reverse proxy.

## Limitations

- Predictions are only reliable within the range of the training data; extreme inputs extrapolate poorly.
- The bundled synthetic data is for demonstration only. Train on real observations (e.g. satellite land surface temperature) for meaningful results.
- If your data is spatial, use a spatial or grouped split (e.g. `GroupKFold`) instead of random splits to avoid overly optimistic scores.
- Load `.joblib` files only from sources you trust, since they can execute code when loaded.

## License

Choose a license for your project (e.g. MIT) and add it here.
