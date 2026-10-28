"""Shared market and ticker catalog for Analytics workbenches."""

from __future__ import annotations

MARKET_CATALOG: dict[str, list[dict[str, str]]] = {
    "US equities": [
        {"ticker": "AAPL", "name": "Apple"},
        {"ticker": "MSFT", "name": "Microsoft"},
        {"ticker": "NVDA", "name": "NVIDIA"},
        {"ticker": "AMZN", "name": "Amazon"},
        {"ticker": "TSLA", "name": "Tesla"},
    ],
    "EU equities": [
        {"ticker": "ASML.AS", "name": "ASML"},
        {"ticker": "SAP.DE", "name": "SAP"},
        {"ticker": "SIE.DE", "name": "Siemens"},
        {"ticker": "LVMH.PA", "name": "LVMH"},
        {"ticker": "AIR.PA", "name": "Airbus"},
    ],
    "UK equities": [
        {"ticker": "AZN.L", "name": "AstraZeneca"},
        {"ticker": "SHEL.L", "name": "Shell"},
        {"ticker": "HSBA.L", "name": "HSBC"},
        {"ticker": "ULVR.L", "name": "Unilever"},
        {"ticker": "LLOY.L", "name": "Lloyds Banking Group"},
    ],
    "JP equities": [
        {"ticker": "7203.T", "name": "Toyota"},
        {"ticker": "6758.T", "name": "Sony"},
        {"ticker": "9984.T", "name": "SoftBank Group"},
        {"ticker": "6861.T", "name": "Keyence"},
        {"ticker": "8306.T", "name": "Mitsubishi UFJ"},
    ],
    "HK equities": [
        {"ticker": "0700.HK", "name": "Tencent"},
        {"ticker": "9988.HK", "name": "Alibaba"},
        {"ticker": "0005.HK", "name": "HSBC Holdings"},
        {"ticker": "0941.HK", "name": "China Mobile"},
        {"ticker": "1299.HK", "name": "AIA Group"},
    ],
    "IN equities": [
        {"ticker": "RELIANCE.NS", "name": "Reliance Industries"},
        {"ticker": "TCS.NS", "name": "Tata Consultancy Services"},
        {"ticker": "INFY.NS", "name": "Infosys"},
        {"ticker": "HDFCBANK.NS", "name": "HDFC Bank"},
        {"ticker": "ICICIBANK.NS", "name": "ICICI Bank"},
    ],
    "ETFs": [
        {"ticker": "SPY", "name": "SPDR S&P 500 ETF"},
        {"ticker": "QQQ", "name": "Invesco QQQ"},
        {"ticker": "IWM", "name": "iShares Russell 2000"},
        {"ticker": "EFA", "name": "iShares MSCI EAFE"},
        {"ticker": "GLD", "name": "SPDR Gold Shares"},
    ],
    "Crypto": [
        {"ticker": "BTC-USD", "name": "Bitcoin"},
        {"ticker": "ETH-USD", "name": "Ethereum"},
        {"ticker": "SOL-USD", "name": "Solana"},
        {"ticker": "BNB-USD", "name": "BNB"},
        {"ticker": "XRP-USD", "name": "XRP"},
    ],
    "Forex": [
        {"ticker": "EURUSD=X", "name": "Euro / US Dollar"},
        {"ticker": "GBPUSD=X", "name": "British Pound / US Dollar"},
        {"ticker": "USDJPY=X", "name": "US Dollar / Japanese Yen"},
        {"ticker": "USDZAR=X", "name": "US Dollar / South African Rand"},
        {"ticker": "AUDUSD=X", "name": "Australian Dollar / US Dollar"},
    ],
    "Commodities": [
        {"ticker": "GC=F", "name": "Gold Futures"},
        {"ticker": "SI=F", "name": "Silver Futures"},
        {"ticker": "CL=F", "name": "WTI Crude Oil Futures"},
        {"ticker": "NG=F", "name": "Natural Gas Futures"},
        {"ticker": "HG=F", "name": "Copper Futures"},
    ],
}


def market_groups() -> list[str]:
    """Return market groups in display order."""
    return list(MARKET_CATALOG)


def tickers_for(group: str) -> list[dict[str, str]]:
    """Return catalog entries for a market group."""
    return MARKET_CATALOG.get(group, MARKET_CATALOG["US equities"])


def catalog_name(ticker: str) -> str:
    """Return the friendly catalog name for a ticker when available."""
    for entries in MARKET_CATALOG.values():
        for entry in entries:
            if entry["ticker"] == ticker:
                return entry["name"]
    return ticker