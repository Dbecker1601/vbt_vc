# Aufgabenliste – vbt_vc Trading Platform

> **Stand:** 2026-03-08 – überarbeitet nach Projektzieldefinition
> **Ziel:** Produktives Trading-System für Futures, Krypto und Forex mit RL-Agenten

---

## Status-Legende

| Symbol | Bedeutung |
|--------|-----------|
| ✅ | Fertig – implementiert und getestet |
| 🔄 | In Arbeit |
| ⬜ | Offen – noch nicht begonnen |
| ❌ | Blockiert oder verworfen |

---

## Phase 1 – Kern-Environment ✅

| # | Aufgabe | Status | Datei(en) |
|---|---------|--------|-----------|
| 1.1 | Abstrakte Basisklasse `TradingEnv` mit Gymnasium-API | ✅ | `trading_env/trading_env.py` |
| 1.2 | State Machine: 3 Positionen, 5 Aktionen, `transform()`-Funktion | ✅ | `trading_env/trading_env.py` |
| 1.3 | Logarithmische Reward-Funktion (Long- und Short-Zyklus) | ✅ | `trading_env/trading_env.py` |
| 1.4 | Observation Space (fensterbasiert, z-Score, + Position + Tick-Ratio) | ✅ | `trading_env/trading_env.py` |
| 1.5 | `render_all()` Visualisierung im AminHP-Stil (2 Panels) | ✅ | `trading_env/trading_env.py` |
| 1.6 | Konkrete `StocksEnv`-Klasse (3 Features: Close, Diff, Volume) | ✅ | `trading_env/stocks_env.py` |
| 1.7 | Greedy-Algorithmus für theoretischen Maximalgewinn | ✅ | `trading_env/stocks_env.py` |

---

## Phase 2 – Datenschicht (`data/`) 🔄

| # | Aufgabe | Status | Datei(en) |
|---|---------|--------|-----------|
| 2.1 | Abstrakte Basisklasse `BaseFetcher` | ✅ | `data/base_fetcher.py` |
| 2.2 | `CCXTFetcher`: OHLCV-Daten via ccxt (Krypto, Futures, Forex) | ✅ | `data/ccxt_fetcher.py` |
| 2.3 | Lokales Datei-Caching (Parquet) | ✅ | `data/ccxt_fetcher.py` |
| 2.4 | Tests für `CCXTFetcher` (Mock, kein echter API-Call) | ✅ | `tests/test_ccxt_fetcher.py` |
| 2.5 | `YFinanceFetcher` als Fallback für Aktien/ETFs | ⬜ | `data/yfinance_fetcher.py` |
| 2.6 | Unterstützung mehrerer Timeframes gleichzeitig (Multi-Timeframe) | ⬜ | `data/` |
| 2.7 | Daten-Validierung & Fehlende-Werte-Behandlung | ⬜ | `data/base_fetcher.py` |

---

## Phase 3 – Backtesting (`backtest/`)

| # | Aufgabe | Status | Datei(en) |
|---|---------|--------|-----------|
| 3.1 | vectorbt-Integration: RL-Agent-Signale → vbt-Portfolio | ⬜ | `backtest/vbt_backtest.py` |
| 3.2 | Konfigurierbare Transaktionskosten & Slippage | ⬜ | `backtest/vbt_backtest.py` |
| 3.3 | Performance-Metriken: Sharpe, Max-Drawdown, Win-Rate, Return | ⬜ | `backtest/metrics.py` |
| 3.4 | Vergleich: RL-Agent vs. Buy-and-Hold | ⬜ | `backtest/vbt_backtest.py` |
| 3.5 | Backtesting-Runner-Skript (`backtest_run.py`) | ⬜ | `backtest_run.py` |
| 3.6 | Tests für Backtesting-Modul | ⬜ | `tests/test_backtest.py` |

---

## Phase 4 – RL-Training (Erweiterung)

| # | Aufgabe | Status | Datei(en) |
|---|---------|--------|-----------|
| 4.1 | `train.py` auf neuen `CCXTFetcher` umstellen | ⬜ | `train.py` |
| 4.2 | Universelles Environment (`UniversalEnv`) für alle Asset-Klassen | ⬜ | `trading_env/universal_env.py` |
| 4.3 | Leverage-Parameter im Environment | ⬜ | `trading_env/trading_env.py` |
| 4.4 | Rollover-Kosten für Futures in Reward-Funktion | ⬜ | `trading_env/trading_env.py` |
| 4.5 | Modell-Checkpoints während Training | ⬜ | `train.py` |
| 4.6 | Modell laden und Weitertraining (`--load-model`) | ⬜ | `train.py` |
| 4.7 | Hyperparameter-Optimierung via Optuna | ⬜ | `train.py` / `agent/` |
| 4.8 | Weitere Algorithmen: SAC, A2C (via SB3) | ⬜ | `train.py` |

---

## Phase 5 – Live-Trading (`live/`)

| # | Aufgabe | Status | Datei(en) |
|---|---------|--------|-----------|
| 5.1 | Abstrakte `BrokerBase`-Klasse | ⬜ | `live/broker_base.py` |
| 5.2 | ccxt-Broker-Implementierung | ⬜ | `live/ccxt_broker.py` |
| 5.3 | Paper-Trading-Modus (Standard – kein echtes Geld) | ⬜ | `live/paper_broker.py` |
| 5.4 | Order-Management (Market, Limit, Stop-Loss) | ⬜ | `live/ccxt_broker.py` |
| 5.5 | Portfolio- und Positions-Tracking | ⬜ | `live/portfolio.py` |
| 5.6 | Risikomanagement (max. Kapital pro Trade, Daily-Stop-Loss) | ⬜ | `live/risk_manager.py` |
| 5.7 | Live-Trading-Loop (`live_run.py`) | ⬜ | `live_run.py` |
| 5.8 | Tests für Live-Trading (vollständig gemockt) | ⬜ | `tests/test_live.py` |

---

## Phase 6 – Infrastruktur & Sicherheit

| # | Aufgabe | Status | Datei(en) |
|---|---------|--------|-----------|
| 6.1 | `.env`-Unterstützung für API-Keys (python-dotenv) | ⬜ | `.env.example`, `config.py` |
| 6.2 | `.env` in `.gitignore` | ⬜ | `.gitignore` |
| 6.3 | YAML-Konfigurationsdateien (Exchange, Symbol, Strategie) | ⬜ | `config/` |
| 6.4 | CI/CD-Pipeline (GitHub Actions – Tests bei Push) | ⬜ | `.github/workflows/` |
| 6.5 | Docker-Container für reproduzierbare Umgebung | ⬜ | `Dockerfile` |

---

## Phase 7 – Dokumentation

| # | Aufgabe | Status | Datei(en) |
|---|---------|--------|-----------|
| 7.1 | `README.md` aktualisieren (neue Architektur, Quick-Start) | ⬜ | `README.md` |
| 7.2 | `ARCHITECTURE.md` – Modulübersicht, Datenfluss | ✅ | `ARCHITECTURE.md` |
| 7.3 | `REQUIREMENTS.md` – Anforderungen (aktualisiert) | ✅ | `REQUIREMENTS.md` |
| 7.4 | `TASKS.md` – Aufgabenliste (aktualisiert) | ✅ | `TASKS.md` |
| 7.5 | Docstrings für alle öffentlichen Klassen und Methoden | ⬜ | alle `*.py` |
| 7.6 | Jupyter-Notebook: Backtesting-Demo | ⬜ | `examples/backtest_demo.ipynb` |
| 7.7 | Jupyter-Notebook: Training-Demo | ⬜ | `examples/training_demo.ipynb` |

---

## Nächste Prioritäten (Reihenfolge)

```
1. ✅ Phase 2 – Datenschicht (CCXTFetcher)       ← gerade in Arbeit
2. ⬜ Phase 3 – vectorbt Backtesting             ← nächster Schritt
3. ⬜ Phase 4 – Environment & Training anpassen
4. ⬜ Phase 6 – Sicherheit (.env, .gitignore)    ← parallel möglich
5. ⬜ Phase 5 – Live-Trading                     ← erst nach Backtesting-Validierung
```

---

## Bekannte Probleme / Technische Schulden

| # | Problem | Schwere | Beschreibung |
|---|---------|---------|--------------|
| B-01 | `StocksEnv` nur für Aktien via yfinance | Hoch | Muss auf universelles OHLCV-Format umgestellt werden |
| B-02 | Kein Leverage / Margin in Reward-Funktion | Hoch | Für Futures-Trading zwingend erforderlich |
| B-03 | Keine Modell-Checkpoints | Mittel | Training-Abbruch verliert Fortschritt |
| B-04 | Keine CI/CD | Mittel | Tests laufen nur lokal |
| B-05 | Fehlende Docstrings | Niedrig | Öffentliche API nicht vollständig dokumentiert |
