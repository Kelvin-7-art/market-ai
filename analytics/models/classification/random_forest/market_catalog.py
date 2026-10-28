"""Market and ticker catalog used by the Random Forest workbenches.

Provides curated lists of supported markets, exchanges, and tickers so the
UI can present a consistent picker across the base and advanced pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Market:
    """A logical market grouping with a curated list of tickers."""

    key: str
    label: str
    region: str
    default_ticker: str
    tickers: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


SUPPORTED_MARKETS: tuple[Market, ...] = (
    Market(
        key="us_equities",
        label="US Equities",
        region="North America",
        default_ticker="AAPL",
        description="Large-cap US stocks and technology leaders.",
        tickers=(
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
            "TSLA", "AMD", "NFLX", "JPM", "BAC", "V", "MA",
            "WMT", "DIS", "INTC", "ORCL", "CRM", "ADBE", "AVGO",
        ),
    ),
    Market(
        key="us_etfs",
        label="US ETFs",
        region="North America",
        default_ticker="SPY",
        description="Broad-market, sector, and thematic ETFs.",
        tickers=(
            "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI",
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
            "ARKK", "SOXX", "SMH", "TLT", "GLD", "SLV",
        ),
    ),
    Market(
        key="uk_equities",
        label="UK Equities",
        region="Europe",
        default_ticker="HSBA.L",
        description="London Stock Exchange listings (Yahoo suffix `.L`).",
        tickers=(
            "HSBA.L", "BP.L", "SHEL.L", "AZN.L", "ULVR.L",
            "GSK.L", "BARC.L", "LLOY.L", "VOD.L", "RIO.L",
            "BHP.L", "DGE.L", "TSCO.L", "NG.L", "REL.L",
        ),
    ),
    Market(
        key="eu_equities",
        label="EU Equities",
        region="Europe",
        default_ticker="ASML.AS",
        description="Euronext, XETRA, and other EU listings.",
        tickers=(
            "ASML.AS", "ADYEN.AS", "PHIA.AS", "HEIA.AS",
            "SAP.DE", "SIE.DE", "BMW.DE", "MBG.DE", "VOW3.DE",
            "AIR.PA", "MC.PA", "OR.PA", "SAN.PA", "BNP.PA",
        ),
    ),
    Market(
        key="jp_equities",
        label="Japan Equities",
        region="Asia",
        default_ticker="7203.T",
        description="Tokyo Stock Exchange listings (Yahoo suffix `.T`).",
        tickers=(
            "7203.T", "6758.T", "6861.T", "9984.T", "8035.T",
            "8306.T", "9432.T", "7974.T", "6098.T", "4063.T",
        ),
    ),
    Market(
        key="hk_equities",
        label="Hong Kong Equities",
        region="Asia",
        default_ticker="0700.HK",
        description="Hong Kong Stock Exchange listings (Yahoo suffix `.HK`).",
        tickers=(
            "0700.HK", "0005.HK", "0939.HK", "3690.HK", "9988.HK",
            "1299.HK", "0388.HK", "2318.HK", "1810.HK", "9618.HK",
        ),
    ),
    Market(
        key="in_equities",
        label="India Equities",
        region="Asia",
        default_ticker="RELIANCE.NS",
        description="NSE India listings (Yahoo suffix `.NS`).",
        tickers=(
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
            "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS",
            "BHARTIARTL.NS", "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        ),
    ),
    Market(
        key="crypto",
        label="Crypto",
        region="Global",
        default_ticker="BTC-USD",
        description="Major cryptocurrencies quoted in USD.",
        tickers=(
            "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
            "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
            "DOT-USD", "LINK-USD", "MATIC-USD", "LTC-USD",
        ),
    ),
    Market(
        key="forex",
        label="Forex",
        region="Global",
        default_ticker="EURUSD=X",
        description="Major FX pairs quoted against USD.",
        tickers=(
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
            "USDCAD=X", "USDCHF=X", "NZDUSD=X",
            "EURGBP=X", "EURJPY=X", "GBPJPY=X",
        ),
    ),
    Market(
        key="commodities",
        label="Commodities",
        region="Global",
        default_ticker="GC=F",
        description="Futures-style commodity quotes from Yahoo Finance.",
        tickers=(
            "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F",
            "HG=F", "ZC=F", "ZS=F", "ZW=F", "PL=F",
        ),
    ),
)


MARKET_BY_KEY: dict[str, Market] = {market.key: market for market in SUPPORTED_MARKETS}


def get_market(key: str) -> Market:
    """Return a market by key, falling back to the first entry."""
    return MARKET_BY_KEY.get(key, SUPPORTED_MARKETS[0])


def all_tickers() -> list[str]:
    """Flat list of every supported ticker (unique, sorted)."""
    seen: set[str] = set()
    out: list[str] = []
    for market in SUPPORTED_MARKETS:
        for ticker in market.tickers:
            if ticker not in seen:
                seen.add(ticker)
                out.append(ticker)
    return out


__all__ = [
    "Market",
    "SUPPORTED_MARKETS",
    "MARKET_BY_KEY",
    "get_market",
    "all_tickers",
]