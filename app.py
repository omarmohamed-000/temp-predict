#!/usr/bin/env python3
"""
Urban Heat Island - Surface Temperature Predictor (Gradio app)

Loads the joblib bundle produced by urban_temp_model.py and serves an
interactive UI with synced sliders + number boxes for each feature.

Run:
    python app.py
    python app.py --model models/urban_temp_model.joblib --port 7860 --share
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("uhi_app")

DEFAULT_MODEL_PATH = "models/urban_temp_model.joblib"

# label, min, max, default, step, help text
FEATURE_SPECS: dict[str, dict] = {
    "green_space_ratio": dict(label="Green space ratio", min=0.0, max=1.0, value=0.25, step=0.01,
                              info="Fraction of area covered by vegetation (0-1)"),
    "building_density": dict(label="Building density", min=0.0, max=1.0, value=0.50, step=0.01,
                             info="Fraction of area covered by buildings (0-1)"),
    "wind_speed": dict(label="Wind speed (m/s)", min=0.0, max=20.0, value=3.0, step=0.1,
                       info="Mean near-surface wind speed"),
    "albedo": dict(label="Surface albedo", min=0.0, max=1.0, value=0.20, step=0.01,
                   info="Surface reflectivity (0 = absorbs all, 1 = reflects all)"),
    "air_temperature": dict(label="Air temperature (°C)", min=-10.0, max=50.0, value=30.0, step=0.1,
                            info="Ambient air temperature"),
    "humidity": dict(label="Relative humidity (%)", min=0.0, max=100.0, value=55.0, step=1.0,
                     info="Relative humidity"),
    "elevation": dict(label="Elevation (m)", min=0.0, max=1000.0, value=100.0, step=1.0,
                      info="Height above sea level"),
    "distance_to_water": dict(label="Distance to water (m)", min=0.0, max=10000.0, value=1500.0, step=10.0,
                              info="Distance to nearest water body"),
    "sky_view_factor": dict(label="Sky view factor", min=0.0, max=1.0, value=0.6, step=0.01,
                            info="Visible sky fraction (0 = fully enclosed, 1 = open)"),
}
FALLBACK_SPEC = dict(min=0.0, max=100.0, value=0.0, step=0.1, info="")


def load_bundle(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model file not found: {p}\n"
            "Train it first with:  python urban_temp_model.py"
        )
    bundle = joblib.load(p)
    for key in ("pipeline", "features"):
        if key not in bundle:
            raise KeyError(f"Model bundle is missing '{key}'. Re-train with urban_temp_model.py.")
    log.info("Loaded model from %s with features: %s", p, bundle["features"])
    return bundle


def temperature_category(t: float) -> tuple[str, str]:
    """Return (label, hex colour) for a surface temperature in °C."""
    if t < 25:
        return "Mild", "#2e8b57"
    if t < 32:
        return "Warm", "#e0a100"
    if t < 38:
        return "Hot", "#e8590c"
    return "Extreme heat", "#c92a2a"


def build_app(bundle: dict) -> gr.Blocks:
    pipeline = bundle["pipeline"]
    features: list[str] = bundle["features"]
    metrics = bundle.get("metrics", {})

    def predict(*values: float) -> str:
        if any(v is None for v in values):
            return "<div style='color:#c92a2a'>Please fill in all inputs.</div>"
        try:
            row = pd.DataFrame([dict(zip(features, values))], columns=features)
            temp = float(pipeline.predict(row)[0])
        except Exception as exc:  # surface any model error to the user cleanly
            log.exception("Prediction failed")
            return f"<div style='color:#c92a2a'>Prediction failed: {exc}</div>"

        label, colour = temperature_category(temp)
        rmse = metrics.get("test_rmse")
        band = (
            f"<div style='opacity:.7;margin-top:6px'>Typical model error: ± {rmse:.2f} °C (RMSE)</div>"
            if rmse is not None else ""
        )
        return (
            f"<div style='text-align:center;padding:24px;border-radius:14px;"
            f"border:2px solid {colour};'>"
            f"<div style='font-size:15px;opacity:.7'>Predicted surface temperature</div>"
            f"<div style='font-size:56px;font-weight:700;color:{colour};line-height:1.2'>{temp:.1f} °C</div>"
            f"<div style='font-size:18px;font-weight:600;color:{colour}'>{label}</div>"
            f"{band}</div>"
        )

    with gr.Blocks(theme=gr.themes.Soft(), title="Urban Heat Island Predictor") as demo:
        gr.Markdown(
            "# 🌡️ Urban Heat Island Temperature Predictor\n"
            "Adjust the climate and geospatial inputs, then click **Predict** "
            "to estimate urban surface temperature."
        )

        sliders: list[gr.Slider] = []
        numbers: list[gr.Number] = []

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                for name in features:
                    spec = {**FALLBACK_SPEC, "label": name.replace("_", " ").capitalize(),
                            **FEATURE_SPECS.get(name, {})}
                    with gr.Row():
                        s = gr.Slider(minimum=spec["min"], maximum=spec["max"], value=spec["value"],
                                      step=spec["step"], label=spec["label"], info=spec["info"],
                                      scale=4)
                        n = gr.Number(value=spec["value"], minimum=spec["min"], maximum=spec["max"],
                                      step=spec["step"], label="Value", scale=1)
                    # Keep slider and number box in sync (user-driven events only)
                    s.input(lambda v: v, inputs=s, outputs=n, show_progress="hidden")
                    n.input(lambda v: v, inputs=n, outputs=s, show_progress="hidden")
                    sliders.append(s)
                    numbers.append(n)

                with gr.Row():
                    reset_btn = gr.Button("Reset", variant="secondary")
                    predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column(scale=2):
                output = gr.HTML(
                    "<div style='text-align:center;padding:24px;opacity:.6'>"
                    "Click <b>Predict</b> to see the result.</div>"
                )
                if metrics:
                    gr.Markdown(
                        "**Model performance (hold-out)**  \n"
                        f"RMSE: `{metrics.get('test_rmse', float('nan')):.3f} °C` · "
                        f"R²: `{metrics.get('test_r2', float('nan')):.3f}`"
                    )

        # The number boxes are the source of truth (they are synced with the sliders)
        predict_btn.click(predict, inputs=numbers, outputs=output)

        def reset():
            vals = [{**FALLBACK_SPEC, **FEATURE_SPECS.get(f, {})}["value"] for f in features]
            return vals + vals

        reset_btn.click(reset, inputs=None, outputs=sliders + numbers)

        gr.Markdown(
            "<sub>Predictions are statistical estimates from a trained XGBoost model and are "
            "only reliable within the range of the training data.</sub>"
        )
    return demo


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Launch the UHI temperature predictor.")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_app(load_bundle(args.model))
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
