# Aufgabenliste – vbt_vc Stock Trading RL Environment

Dieses Dokument beschreibt was bereits umgesetzt wurde und was noch zu tun ist.

---

## Status-Legende

| Symbol | Bedeutung |
|--------|-----------|
| ✅ | Fertig – implementiert und getestet |
| 🔄 | In Arbeit |
| ⬜ | Offen – noch nicht begonnen |
| ❌ | Blockiert oder verworfen |

---

## Phase 1 – Kern-Environment

| # | Aufgabe | Status | Datei(en) |
|---|---------|--------|-----------|
| 1.1 | Abstrakte Basisklasse `TradingEnv` mit Gymnasium-API | ✅ | `trading_env/trading_env.py` |
| 1.2 | State Machine: 3 Positionen, 5 Aktionen, `transform()`-Funktion | ✅ | `trading_env/trading_env.py` |
| 1.3 | Logarithmische Reward-Funktion (Long- und Short-Zyklus) | ✅ | `trading_env/trading_env.py` |
| 1.4 | Observation Space (fensterbasiert, z-Score normalisiert, + Position + Tick-Ratio) | ✅ | `trading_env/trading_env.py` |
| 1.5 | `render_all()` Visualisierung im AminHP-Stil (2 Panels, Farbmarkierungen) | ✅ | `trading_env/trading_env.py` |
| 1.6 | Konkrete `StocksEnv`-Klasse (3 Features: Close, Diff, Volume) | ✅ | `trading_env/stocks_env.py` |
| 1.7 | Greedy-Algorithmus zur Berechnung des theoretischen Maximalgewinns | ✅ | `trading_env/stocks_env.py` |
| 1.8 | Package-Exports (`__init__.py`) | ✅ | `trading_env/__init__.py` |

---

## Phase 2 – Training & Evaluation

| # | Aufgabe | Status | Datei(en) |
|---|---------|--------|-----------|
| 2.1 | End-to-End-Trainingsskript mit PPO (Stable Baselines 3) | ✅ | `train.py` |
| 2.2 | Automatischer Daten-Download via `yfinance` | ✅ | `train.py` |
| 2.3 | Train/Test-Split per konfigurierbarem Datum | ✅ | `train.py` |
| 2.4 | SB3 `check_env()`-Validierung | ✅ | `train.py` |
| 2.5 | TensorBoard-Logging | ✅ | `train.py` |
| 2.6 | CLI-Argumente (`--ticker`, `--timesteps`, `--window`, usw.) | ✅ | `train.py` |
| 2.7 | Modellspeicherung nach Training | ⬜ | `train.py` |
| 2.8 | Modell-Laden und Weitertraining (`--load-model`-Flag) | ⬜ | `train.py` |

---

## Phase 3 – Tests

| # | Aufgabe | Status | Datei(en) |
|---|---------|--------|-----------|
| 3.1 | State-Machine-Tests (alle Übergänge) | ✅ | `tests/test_trading_env.py` |
| 3.2 | Gymnasium-API-Konformitätstests | ✅ | `tests/test_trading_env.py` |
| 3.3 | Reward-Berechnungstests | ✅ | `tests/test_trading_env.py` |
| 3.4 | Visualisierungs-Test (`render_all()` → PNG) | ✅ | `tests/test_trading_env.py` |
| 3.5 | SB3 `check_env()`-Test | ✅ | `tests/test_trading_env.py` |
| 3.6 | Tests für `StocksEnv` mit echten yfinance-Daten | ⬜ | `tests/` |
| 3.7 | Edge-Case-Tests (leere Daten, ungültige Frames, etc.) | ⬜ | `tests/` |
| 3.8 | Integrations-/End-to-End-Test für `train.py` | ⬜ | `tests/` |

---

## Phase 4 – Dokumentation

| # | Aufgabe | Status | Datei(en) |
|---|---------|--------|-----------|
| 4.1 | `README.md` mit Architekturübersicht, Quick-Start, CLI-Referenz | ✅ | `README.md` |
| 4.2 | `AGENT_ROLES.md` – LLM Developer/Kritiker-Workflow (Deutsch) | ✅ | `AGENT_ROLES.md` |
| 4.3 | `REQUIREMENTS.md` – funktionale & nicht-funktionale Anforderungen | ✅ | `REQUIREMENTS.md` |
| 4.4 | `TASKS.md` – Aufgabenliste mit Status | ✅ | `TASKS.md` |
| 4.5 | Inline-Docstrings für alle öffentlichen Klassen und Methoden | ⬜ | `trading_env/*.py` |
| 4.6 | CHANGELOG.md – Versionshistorie | ⬜ | `CHANGELOG.md` |
| 4.7 | Beispiel-Notebooks (Jupyter) für Training und Analyse | ⬜ | `examples/` |

---

## Phase 5 – Erweiterungen (optional / Zukunft)

| # | Aufgabe | Status | Beschreibung |
|---|---------|--------|--------------|
| 5.1 | Weitere RL-Algorithmen (A2C, SAC, TD3) | ⬜ | Über SB3-Parameter steuerbar |
| 5.2 | Erweiterter Observation Space (technische Indikatoren) | ⬜ | RSI, MACD, Bollinger Bands, usw. |
| 5.3 | Multi-Asset-Environment | ⬜ | Gleichzeitiges Trading mehrerer Ticker |
| 5.4 | Portfolio-Management-Logik | ⬜ | Gewichtung und Rebalancing |
| 5.5 | Live-Trading-Anbindung (Paper Trading) | ⬜ | Alpaca, Interactive Brokers API |
| 5.6 | Hyperparameter-Optimierung (Optuna) | ⬜ | Automatisches Tuning von PPO-Parametern |
| 5.7 | CI/CD-Pipeline (GitHub Actions) | ⬜ | Automatische Tests bei jedem Push |
| 5.8 | Docker-Container für reproduzierbare Umgebung | ⬜ | `Dockerfile` + `docker-compose.yml` |

---

## Nächste Prioritäten

1. **⬜ 2.7** – Modellspeicherung implementieren (`train.py`)
2. **⬜ 2.8** – Modell-Laden-Feature (`--load-model`)
3. **⬜ 3.6** – Tests mit echten yfinance-Daten
4. **⬜ 3.7** – Edge-Case-Tests
5. **⬜ 4.5** – Docstrings vervollständigen
6. **⬜ 5.7** – CI/CD-Pipeline (GitHub Actions)

---

## Bekannte Probleme / Technische Schulden

| # | Problem | Schwere | Beschreibung |
|---|---------|---------|--------------|
| B-01 | Kein Modell-Checkpoint-System | Mittel | Modelle werden nach dem Training nicht automatisch gespeichert |
| B-02 | Keine explizite Fehlerbehandlung bei ungültigen Daten | Niedrig | Fehler bei leerem DataFrame erst zur Laufzeit |
| B-03 | Fehlende Docstrings | Niedrig | Öffentliche API nicht vollständig dokumentiert |
| B-04 | Keine CI/CD | Mittel | Tests laufen nur lokal, kein automatischer Check bei Push |
