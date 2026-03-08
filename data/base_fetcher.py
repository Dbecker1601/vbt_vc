"""Abstract base class for all market data fetchers."""

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


CACHE_DIR = Path(__file__).parent / "cache"


class BaseFetcher(ABC):
    """Common interface for all data sources.

    Concrete implementations must return a DataFrame with columns:
    ``Open``, ``High``, ``Low``, ``Close``, ``Volume``
    indexed by a timezone-aware or naive UTC DatetimeIndex.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: str | None = None,
        until: str | None = None,
        cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV data, optionally using a local Parquet cache.

        Parameters
        ----------
        symbol:
            Exchange symbol, e.g. ``"BTC/USDT"`` or ``"BTC/USDT:USDT"``
            (perpetual future).
        timeframe:
            Candle duration string accepted by the underlying source,
            e.g. ``"1m"``, ``"1h"``, ``"1d"``.
        since:
            Start date as ISO-8601 string (``"YYYY-MM-DD"``).
            ``None`` fetches from the earliest available data.
        until:
            End date as ISO-8601 string (``"YYYY-MM-DD"``).
            ``None`` fetches up to the most recent data.
        cache:
            If ``True``, results are read from / written to a local
            Parquet file to avoid redundant API calls.

        Returns
        -------
        pd.DataFrame
            Columns: ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.
            Index: ``pd.DatetimeIndex`` (UTC).
        """
        if cache:
            path = self._cache_path(symbol, timeframe, since, until)
            if path.exists():
                return pd.read_parquet(path)

        df = self._fetch(symbol, timeframe, since, until)
        df = self._validate(df)

        if cache:
            df.to_parquet(path)

        return df

    # ------------------------------------------------------------------
    # Abstract methods – implement in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _fetch(
        self,
        symbol: str,
        timeframe: str,
        since: str | None,
        until: str | None,
    ) -> pd.DataFrame:
        """Fetch raw OHLCV data from the underlying source."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cache_path(
        self,
        symbol: str,
        timeframe: str,
        since: str | None,
        until: str | None,
    ) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(":", "-")
        since_s = since or "start"
        until_s = until or "end"
        name = f"{self._source_id()}_{safe_symbol}_{timeframe}_{since_s}_{until_s}.parquet"
        return self.cache_dir / name

    @abstractmethod
    def _source_id(self) -> str:
        """Short identifier used in cache file names, e.g. ``'binance'``."""

    @staticmethod
    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        if df.empty:
            raise ValueError("Fetched DataFrame is empty.")
        return df.sort_index()
