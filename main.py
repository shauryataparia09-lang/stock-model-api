"""
main.py
=======
FastAPI backend that exposes the stock model over HTTP, so a Lovable
(or any other) frontend can call it directly with fetch().

Endpoints:
  GET  /                          -> health check
  GET  /analyze?ticker=AAPL       -> single stock prediction
  GET  /compare?tickers=AAPL,MSFT,TSLA  -> multi-stock comparison

Run locally:
    uvicorn main:app --reload --port 8000

Then test in a browser or curl:
    http://localhost:8000/analyze?ticker=AAPL
    http://localhost:8000/compare?tickers=AAPL,MSFT,TSLA

See README.md for how to deploy this so Lovable can reach it over the
internet (a local server only works while your own computer is running it).
"""

from dataclasses import asdict
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from model import analyze_stock, compare_stocks

app = FastAPI(
    title="Stock Technical Analysis API",
    description="Predicts likely next-month price direction for stocks using auto-selected technical indicators.",
    version="1.0.0",
)

# Allow requests from any origin so a Lovable-hosted frontend can call this
# freely. If you want to lock this down later, replace "*" with your
# Lovable site's exact URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Stock Technical Analysis API is running."}


@app.get("/analyze")
def analyze(ticker: str = Query(..., description="Stock ticker, e.g. AAPL, TSLA, INFY.NS")):
    result = analyze_stock(ticker)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return asdict(result)


@app.get("/compare")
def compare(
    tickers: str = Query(..., description="Comma-separated tickers, e.g. AAPL,MSFT,TSLA")
):
    ticker_list: List[str] = [t.strip() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    if len(ticker_list) > 10:
        raise HTTPException(status_code=400, detail="Please compare 10 tickers or fewer at a time.")

    results = compare_stocks(ticker_list)
    return {"results": [asdict(r) for r in results]}
