# EVOO Price Forecasting — Bari Wholesale Market

A machine learning project to forecast Extra Virgin Olive Oil wholesale prices using weekly data from the Camera di Commercio di Bari, augmented with weather, financial, and harvest signals.

**Spoiler:** the models don't beat a naive benchmark. That's the finding.

---

## Project Overview

This project attempts to predict short-term weekly percentage price changes for Extra Vergine di Oliva (acidity ≤ 0.4%) at the Bari wholesale market. Itis documented in full on paean.net

**Not for trading.**

---

## Market and Data Scope

| Parameter | Value |
|---|---|
| Country | Italy |
| Market hub | Bari |
| Price source | Camera di Commercio di Bari |
| Product | Extra Vergine di Oliva, acidity ≤ 0.4% |
| Price unit | €/kg |
| Frequency | Weekly |
| Coverage | September 2020 – March 2026 |
| Observations | 289 weekly rows |

Prices are published as a Min/Max range. The target variable is the weekly percentage change of the midpoint: `(Min + Max) / 2`.

---

## Repository Structure

```
EVOO/
│
├── EVOO_Prices_Bari.csv          # Raw price data (manually extracted from PDFs)
├── EVOO_dataset_final.csv        # Final feature dataset (all signals merged)
├── EVOO_dataset_interim.csv      # Interim save (financial + weather signals)
│
├── 01_load_and_explore.ipynb     # Data loading, cleaning, imputation, first chart
├── price_series.png              # Time series chart (price level + weekly % change)
│
└── README.md
```

---

## Data Sources

| Signal | Source | Method | Frequency |
|---|---|---|---|
| EVOO wholesale prices | Camera di Commercio di Bari | Manual PDF extraction | Weekly |
| EUR/USD exchange rate | FRED (`DEXUSEU`) | API | Daily → Weekly |
| Brent crude oil | FRED (`DCOILBRENTEU`) | API | Daily → Weekly |
| Puglia weather | Open-Meteo (41.1°N, 16.4°E) | API | Daily → Weekly |
| Andalusia weather | Open-Meteo (37.8°N, 3.8°W) | API | Daily → Weekly |
| Google Trends | pytrends | API | Weekly |
| Harvest production | International Olive Council | Manual | Annual |

---

## Feature Engineering

**41 features in total across six groups:**

**Price history:** lagged weekly % changes (1, 2, 3, 4, 8 weeks), 52-week rolling z-score of price level.

**Seasonality:** week of year, binary harvest season flag (October–November).

**Financial signals:** EUR/USD rate, Brent crude price.

**Weather — Puglia and Andalusia:** weekly rainfall, max temperature, evapotranspiration, water balance (rainfall minus evapotranspiration), 4-week and 13-week rolling averages of each, extreme rainfall flag (>40mm/week).

**Sentiment:** Google Trends index for "olio extravergine" (Italian) and "olive oil price" (English), rescaled across two overlapping time chunks to achieve weekly resolution.

**Harvest fundamentals:** Italy and Spain annual production in thousand tonnes (IOC data), z-scored relative to the series mean. Assigned to crop years (November–October).

---

## Models

Four forecast horizons: **1, 2, 4 and 8 weeks ahead.**

Train/test split: **September 2020 – December 2023 (train) / January 2024 – March 2026 (test).**

| Model | Features |
|---|---|
| Naive baseline | Always predict zero change |
| Linear regression | Price history only |
| Random Forest (Layer 1) | Price history + seasonality |
| Random Forest (Layer 2) | All 41 features |
| Gradient Boosting (Layer 1) | Price history + seasonality |
| Random Forest classifier | All 41 features, directional labels |

**Metrics:** MAE, RMSE, directional accuracy (regression); overall accuracy, directional accuracy on moving weeks, classification report (classifier).

---

## Results

### Regression models (1-week horizon)

| Model | MAE | RMSE | Dir% |
|---|---|---|---|
| Naive (predict zero) | 0.708 | 2.088 | 79.8% |
| Linear regression | 0.938 | 2.140 | 9.6% |
| Random Forest (prices only) | 1.044 | 2.069 | 10.5% |
| Random Forest (augmented) | 1.175 | 2.314 | 6.1% |
| Gradient Boosting | 1.615 | 2.391 | 8.8% |

**The naive model won on every metric at every horizon.**

### Classification model (1-week horizon)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Down (< -0.5%) | 0.33 | 0.14 | 0.20 |
| Flat (±0.5%) | 0.84 | 0.80 | 0.82 |
| Up (> +0.5%) | 0.00 | 0.00 | 0.00 |

The classifier correctly identifies flat weeks but cannot reliably predict directional moves.

---

## Key Findings

**1. Past prices do not predict future prices.**
All regression models failed to beat a naive zero-change prediction using price history alone. This is consistent with weak-form market efficiency — available price information is already incorporated into current prices.

**2. External signals have structure but insufficient predictive power at weekly frequency.**
The most important features identified across all models were `precip_mm_13w` (13-week rolling rainfall in Puglia), `eurusd`, and `precip_esp_13w` (13-week rolling rainfall in Andalusia). These are genuine, meaningful signals — but they drive prices over months, not weeks.

**3. The naive benchmark is very hard to beat on this series.**
Over 73% of weekly observations showed a price change of less than 0.5% in either direction. A model that always predicts zero is correct three weeks out of four by construction.

**4. The real signals operate at the wrong timescale.**
A drought in Andalusia in July does not move the Bari price list the following Tuesday. It moves prices gradually over the following harvest season. Weekly ML forecasting is therefore the wrong tool for this market at this frequency.

**5. What the models did learn.**
Feature importances consistently identified accumulated rainfall (13-week rolling), EUR/USD, and price level relative to recent history as the most informative signals — a plausible and interpretable description of how the EVOO market actually works.

---

## Setup and Reproduction

### Requirements

```bash
pip install pandas numpy matplotlib scikit-learn fredapi pytrends requests
```

### API Keys

A free FRED API key is required for EUR/USD and Brent crude data. Register at [fred.stlouisfed.org](https://fred.stlouisfed.org) and add your key to the notebook where indicated. Open-Meteo and Google Trends require no API key.

### Data

The raw price CSV (`EVOO_Prices_Bari.csv`) was extracted manually from weekly PDFs published by the [Camera di Commercio di Bari](https://www.ba.camcom.it). The PDFs are publicly available on their website. The final feature dataset (`EVOO_dataset_final.csv`) is included in this repository and can be used directly to reproduce the modelling results without re-fetching all external data.

### Running the notebooks

Open `01_load_and_explore.ipynb` in Jupyter and run all cells in order. All data fetching, feature engineering, and modelling is contained in a single notebook.

---

## What's Next

The consistent finding that accumulated seasonal signals (13-week rainfall, harvest size) matter more than weekly signals motivates a different modelling approach: predicting the **direction of the seasonal price trend** over a 3–6 month horizon, where these signals are better matched to the forecasting frequency. That is the subject of a follow-up project.

---

## Author

Max — data analyst and ML practitioner.

*Data source: Camera di Commercio di Bari. External signals: FRED, Open-Meteo, Google Trends, International Olive Council.*

