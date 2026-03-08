# Anforderungen – vbt_vc Stock Trading RL Environment

Dieses Dokument beschreibt alle funktionalen und nicht-funktionalen Anforderungen des Projekts.

---

## 1. Projektziel

Entwicklung eines Reinforcement-Learning-Environments für den Aktienhandel, das:

- die **State-Machine-Logik von DI-engine** (OpenDILab) verwendet,
- den **Visualisierungsstil von AminHP** (gym-anytrading) umsetzt,
- vollständig **kompatibel mit Stable Baselines 3** (SB3) ist.

---

## 2. Funktionale Anforderungen

### 2.1 State Machine

| ID | Anforderung |
|----|-------------|
| F-01 | Das Environment muss 3 Positionen unterstützen: **SHORT** (−1), **FLAT** (0), **LONG** (+1). |
| F-02 | Es müssen 5 Aktionen definiert sein: **DOUBLE_SELL** (0), **SELL** (1), **HOLD** (2), **BUY** (3), **DOUBLE_BUY** (4). |
| F-03 | Die Zustandsübergänge müssen der DI-engine-Übergangstabelle entsprechen. |
| F-04 | Ungültige Aktionen (z. B. BUY bei LONG-Position) dürfen keine Exception auslösen, sondern müssen korrekt ignoriert oder behandelt werden. |

### 2.2 Reward-Funktion

| ID | Anforderung |
|----|-------------|
| F-05 | Reward ist nur dann ungleich null, wenn ein vollständiger Trade-Zyklus abgeschlossen wird. |
| F-06 | **Long-Zyklus**: `reward = log(close_curr / close_prev) + log(cost)` |
| F-07 | **Short-Zyklus**: `reward = log(2 − close_curr / close_prev) + log(cost)` |
| F-08 | Der Kostenfaktor ergibt sich aus: `cost = (1 − bid_fee) × (1 − ask_fee)`. |

### 2.3 Observation Space

| ID | Anforderung |
|----|-------------|
| F-09 | Der Beobachtungsvektor muss flach (1D) sein. |
| F-10 | Er enthält `window_size × n_features` normalisierte Feature-Werte (z-Score). |
| F-11 | Zusätzlich werden **Position** (−1 / 0 / +1) und **Tick-Ratio** (ticks seit letztem Trade / Episodenlänge) angehängt. |
| F-12 | Für Aktiendaten: 3 Features – **Close**, **Diff** (Preisdifferenz), **Volume**. |

### 2.4 Gymnasium-Schnittstelle

| ID | Anforderung |
|----|-------------|
| F-13 | Das Environment muss die `gymnasium`-API implementieren (`reset()`, `step()`, `render()`, `close()`). |
| F-14 | Es muss die SB3-Validierung (`check_env()`) fehlerfrei bestehen. |
| F-15 | `reset()` gibt `(observation, info)` zurück. |
| F-16 | `step()` gibt `(observation, reward, terminated, truncated, info)` zurück. |

### 2.5 Dateneingabe

| ID | Anforderung |
|----|-------------|
| F-17 | Eingabedaten sind OHLCV-DataFrames im `pandas`-Format (z. B. von `yfinance`). |
| F-18 | Das Environment muss einen konfigurierbaren `frame_bound`-Parameter unterstützen (Start-/End-Tick). |
| F-19 | Das Environment muss einen konfigurierbaren `window_size`-Parameter unterstützen. |

### 2.6 Visualisierung

| ID | Anforderung |
|----|-------------|
| F-20 | `render_all()` erzeugt ein zweiteiliges Diagramm im AminHP-Stil. |
| F-21 | **Oberes Panel**: Schlusskurs mit farbigen Positionsmarkierungen (▲ grün = Long, ● blau = Flat, ▼ rot = Short). |
| F-22 | **Unteres Panel**: Akkumulierte Profitquote als Kurvendiagramm. |
| F-23 | Der Plot kann optional als Bilddatei (PNG) gespeichert werden (`save_path`-Parameter). |

### 2.7 Training

| ID | Anforderung |
|----|-------------|
| F-24 | Ein CLI-Trainingsskript (`train.py`) muss PPO (Stable Baselines 3) unterstützen. |
| F-25 | Aktienhistorie wird automatisch via `yfinance` heruntergeladen. |
| F-26 | Das Skript trennt Daten in Trainings- und Testmenge anhand eines konfigurierbaren Datums. |
| F-27 | TensorBoard-Logging muss unterstützt werden. |
| F-28 | Nach dem Training wird das Modell evaluiert und ein `render_all()`-Plot gespeichert. |

---

## 3. Nicht-funktionale Anforderungen

| ID | Anforderung |
|----|-------------|
| NF-01 | Alle Komponenten müssen mit **Python ≥ 3.9** kompatibel sein. |
| NF-02 | Abhängigkeiten sind in `requirements.txt` gelistet und versioniert. |
| NF-03 | Der Code muss durch eine **automatisierte Test-Suite** (pytest) abgedeckt sein. |
| NF-04 | Die Testabdeckung soll **alle State-Machine-Übergänge**, die Gymnasium-API, die Reward-Berechnung und die Visualisierung umfassen. |
| NF-05 | Code-Qualität folgt **PEP 8** und Python-Best-Practices. |
| NF-06 | Bei jeder Code-Generierung durch ein LLM ist der **Developer-Kritiker-Workflow** (siehe `AGENT_ROLES.md`) einzuhalten. |
| NF-07 | Das Environment soll erweiterbar sein – neue Environments können durch Ableitung von `TradingEnv` erstellt werden. |

---

## 4. Abhängigkeiten

| Paket | Mindestversion | Zweck |
|-------|---------------|-------|
| `gymnasium` | ≥ 0.26.0 | Standard-RL-Environment-API |
| `stable-baselines3` | ≥ 2.0.0 | PPO und weitere RL-Algorithmen |
| `numpy` | ≥ 1.24.0 | Numerische Berechnungen |
| `pandas` | ≥ 2.0.0 | Datenverarbeitung |
| `yfinance` | ≥ 0.2.30 | Yahoo-Finance-Datenabruf |
| `matplotlib` | ≥ 3.7.0 | Visualisierung |
| `torch` | ≥ 2.0.0 | Neural-Network-Backend für SB3 |

---

## 5. Quellenreferenzen

- **DI-engine State Machine & Reward**: [opendilab/DI-engine – dizoo/gym_anytrading](https://github.com/opendilab/DI-engine/tree/main/dizoo/gym_anytrading/envs)
- **Visualisierung**: [AminHP/gym-anytrading](https://github.com/AminHP/gym-anytrading)
- **RL-Training**: [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)
