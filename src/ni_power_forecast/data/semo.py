from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

STATIC_REPORTS_URL = "https://reports.sem-o.com/api/v1/documents/static-reports"
DOCUMENT_URL = "https://reports.sem-o.com/api/v1/documents/{document_id}"


def _exclusive_end(date_to: str) -> str:
    end = date.fromisoformat(date_to) + timedelta(days=1)
    return end.isoformat()


def list_day_ahead_reports(date_from: str, date_to: str, page_size: int = 500) -> dict[str, Any]:
    """Query the public SEMO report API for SEMOpx day-ahead market-result reports."""
    params = {
        "DPuG_ID": "EA-001",
        "ResourceName": "MarketResult_SEM-DA",
        "Date": f">={date_from}<{_exclusive_end(date_to)}",
        "sort_by": "Date",
        "order_by": "ASC",
        "page_size": page_size,
        "ExcludeDelayedPublication": "0",
    }
    response = requests.get(STATIC_REPORTS_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _fetch_document(document_id: str) -> dict[str, Any]:
    response = requests.get(
        DOCUMENT_URL.format(document_id=document_id), params={"IST": 0}, timeout=30
    )
    response.raise_for_status()
    return response.json()


def download_report_documents(date_from: str, date_to: str, output_dir: str | Path) -> list[Path]:
    """Download raw SEMO JSON documents for reproducible local parsing."""
    listing = list_day_ahead_reports(date_from, date_to)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for item in listing.get("items", []):
        document_id = item.get("_id")
        if not document_id:
            continue
        payload = _fetch_document(document_id)
        path = out / f"{document_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        saved.append(path)
    return saved


def _flatten_scalars(value):
    if isinstance(value, list):
        for item in value:
            yield from _flatten_scalars(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_scalars(item)
    else:
        yield value


def inspect_document(path: str | Path) -> pd.DataFrame:
    """Turn raw report rows into a searchable scalar view for schema inspection."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for i, row in enumerate(payload.get("rows", [])):
        scalars = [x for x in _flatten_scalars(row) if x is not None]
        records.append({"row": i, "values": " | ".join(map(str, scalars))})
    return pd.DataFrame(records)


def parse_market_result_document(
    payload: dict[str, Any], market: str = "NI-DA", currency: str = "GBP"
) -> pd.DataFrame:
    """Extract a price series from an EA-001 market-result JSON document.

    EA-001 market blocks consist of a market identifier followed by repeated
    metadata / timestamps / values triplets. Rather than rely on a fixed column
    position, the parser scores each triplet by its metadata and selects the
    price series matching the requested currency.
    """
    market_block = None
    for block in payload.get("rows", []):
        if not isinstance(block, list) or not block:
            continue
        header = block[0]
        if isinstance(header, list) and len(header) > 1 and str(header[1]) == market:
            market_block = block
            break

    if market_block is None:
        available = []
        for block in payload.get("rows", []):
            if (
                isinstance(block, list)
                and block
                and isinstance(block[0], list)
                and len(block[0]) > 1
            ):
                available.append(str(block[0][1]))
        raise ValueError(
            f"Market {market!r} not found. Available markets: {sorted(set(available))}"
        )

    candidates: list[tuple[int, str, list[Any], list[Any]]] = []
    for i in range(1, len(market_block) - 2, 3):
        metadata, times, values = market_block[i : i + 3]
        if not isinstance(times, list) or not isinstance(values, list) or len(times) != len(values):
            continue
        label = " ".join(str(v) for v in _flatten_scalars(metadata) if v is not None)
        lower = label.lower()
        score = 0
        if "price" in lower:
            score += 4
        if currency.lower() in lower:
            score += 3
        if "index" in lower or "reference" in lower:
            score += 1
        candidates.append((score, label, times, values))

    if not candidates:
        raise ValueError("No timestamp/value series found in market block")

    candidates.sort(key=lambda item: item[0], reverse=True)
    score, label, times, values = candidates[0]
    if score < 4:
        labels = [candidate[1] for candidate in candidates]
        raise ValueError(
            f"Could not identify a {currency} price series for {market}. Series metadata: {labels}"
        )

    timestamp = pd.to_datetime(pd.Series(times), utc=True, errors="coerce")
    numeric = pd.to_numeric(
        pd.Series(values).astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    out = pd.DataFrame({"timestamp": timestamp, "price_gbp_mwh": numeric})
    out = out.dropna().drop_duplicates("timestamp").sort_values("timestamp")
    out["source"] = f"SEMO EA-001 {market} {currency} ({label})"
    return out.reset_index(drop=True)


def parse_market_result_file(
    path: str | Path, market: str = "NI-DA", currency: str = "GBP"
) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_market_result_document(payload, market=market, currency=currency)


def fetch_day_ahead_prices(
    date_from: str,
    date_to: str,
    market: str = "NI-DA",
    currency: str = "GBP",
) -> pd.DataFrame:
    """Fetch and normalise public SEMOpx day-ahead prices for a date range."""
    listing = list_day_ahead_reports(date_from, date_to)
    frames = []
    for item in listing.get("items", []):
        document_id = item.get("_id")
        if not document_id:
            continue
        payload = _fetch_document(document_id)
        try:
            frames.append(parse_market_result_document(payload, market=market, currency=currency))
        except ValueError:
            # Some listings can contain duplicates/revisions that do not expose the requested area.
            continue
    if not frames:
        raise ValueError(
            f"No {market}/{currency} day-ahead price data parsed for {date_from}..{date_to}"
        )
    out = pd.concat(frames, ignore_index=True)
    return (
        out.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
