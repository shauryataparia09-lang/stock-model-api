# Stock Technical Analysis API

This is the "brain" of your finance tool — no website, just the model,
exposed as an API. Lovable will build the website that talks to it.

## What it does

For any stock ticker, it:
1. Pulls ~5 years of price history (via Yahoo Finance).
2. Computes ~15 technical indicators (moving averages, RSI, MACD,
   Bollinger Bands, volatility, volume trends, etc.).
3. Trains a model on that stock's own history to learn which indicators
   actually predicted future moves for it — so different stocks can be
   driven by different indicators, chosen automatically.
4. Predicts whether the price is more likely to be UP or DOWN about a
   month from now, with a confidence level.
5. For `/compare`, does this for several tickers and ranks them.

**This is a statistical model based on historical patterns — not
investment advice, and no model can reliably predict markets.** Treat
outputs as one input among many, not a guarantee.

## Files

- `model.py` — all the analysis/prediction logic (no web code).
- `main.py` — wraps `model.py` as a web API (FastAPI).
- `requirements.txt` — Python packages needed.

## Step 1 — Run it locally first (recommended)

1. Install Python 3: https://www.python.org/downloads/
2. In this folder, run:
   ```
   pip install -r requirements.txt
   ```
3. Start the server:
   ```
   uvicorn main:app --reload --port 8000
   ```
4. Open in a browser:
   ```
   http://localhost:8000/analyze?ticker=AAPL
   http://localhost:8000/compare?tickers=AAPL,MSFT,TSLA
   ```
   You should see JSON with the prediction.

This only works while your own computer is running the server — for
Lovable's website to use it from anywhere, you need to deploy it (next step).

## Step 2 — Deploy it so Lovable can reach it

Lovable's website runs in the browser/cloud, so it can't call a server on
your laptop. You need to host this API somewhere with a public URL. The
easiest free options:

**Render.com (recommended, free tier, simplest):**
1. Push this folder to a GitHub repo.
2. Go to https://render.com → New → Web Service → connect your repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy. Render gives you a public URL like `https://your-app.onrender.com`.

**Railway.app** works the same way and is also beginner-friendly.

Once deployed, test the same URLs as above but with your public domain,
e.g. `https://your-app.onrender.com/analyze?ticker=AAPL`.

## Step 3 — Connect it in Lovable

In your Lovable prompt/chat, tell it something like:

> "Call this API to get stock predictions:
> `GET https://your-app.onrender.com/analyze?ticker={ticker}`
> and `GET https://your-app.onrender.com/compare?tickers={ticker1,ticker2}`.
> Both return JSON — show `predicted_direction`, `probability_up`,
> `confidence`, and `top_indicators` in the UI."

Lovable can call this directly from the frontend with `fetch()`, since
the API already allows requests from any website (CORS is open).

## Example response (`/analyze?ticker=AAPL`)

```json
{
  "ticker": "AAPL",
  "success": true,
  "last_price": 227.5,
  "last_date": "2026-09-18",
  "predicted_direction": "UP",
  "probability_up": 0.63,
  "confidence": "Medium",
  "horizon_days": 21,
  "top_indicators": [
    {"indicator": "Price vs 200-day average", "importance": 0.19},
    {"indicator": "RSI (14-day momentum)", "importance": 0.14}
  ],
  "backtest_accuracy": 0.58,
  "current_indicator_values": { "...": "..." }
}
```

## Notes & limitations

- Needs internet access to fetch live price data (Yahoo Finance).
- Works for most global tickers Yahoo Finance supports (add exchange
  suffixes where needed, e.g. `.NS` for NSE India, `.L` for London).
- `backtest_accuracy` tells you how well the model did on recent unseen
  historical data — use it to judge how much to trust a given prediction.
- The horizon is fixed at ~1 month (21 trading days) per your spec; this
  can be changed in `model.py` (`PREDICTION_HORIZON_DAYS`) if you want to
  experiment with other horizons later.
