# SupeRR Selector Season 2 — Round 1

## Identifying the Most Reliable Players Under Pressure

An advanced, data-driven cricket analytics framework to identify and rank the most reliable T20 players under pressure, built on ball-by-ball match data from IPL and T20 Internationals (2022–2026).

---

## 📌 Overview

Traditional cricket metrics like raw batting average, strike rate, or bowling economy treat all deliveries equally. A six hit when 60 runs are needed off 6 balls receives the same weight as a boundary in a nail-biting final over.

**SupeRR Selector** constructs a granular, delivery-level **Pressure Index (PI)** from three foundational pillars:
1. **Match Stakes Ladder:** Dynamic competition tier weighting (Bilateral & Qualifiers < League matches < Playoffs/Finals).
2. **Opponent Strength:** Dynamic pre-match Elo rating per team, serving as a responsive proxy for team quality and ICC rankings.
3. **Situational Pressure:** A calibrated win-probability framework factoring in target/required run rate, balls remaining, wickets left, and team Elo differential.

Players' performances across the highest-pressure deliveries (top 30% pressure tier) are evaluated against a **level-weighted expectation** (85% neutral situation par vs. 15% player baseline), incorporating Bayesian sample-size shrinkage and innings-level consistency weighting.

---

## 🏆 Headline Rankings

### Top 10 Batters Under Pressure
| Rank | Player | Pressure Score | Pressure Runs | Pressure Balls | Pressure SR | Baseline SR | SR Boost | Consistency |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Shreyas Iyer** | **+0.333** | 684 | 436 | 156.9 | 134.7 | +22.2 | 58.7% |
| **2** | **Suryakumar Yadav** | **+0.330** | 1083 | 647 | 167.4 | 161.4 | +6.0 | 58.8% |
| **3** | **Shubman Gill** | **+0.329** | 983 | 639 | 153.8 | 143.1 | +10.8 | 61.3% |
| **4** | **Rajat Patidar** | **+0.318** | 569 | 338 | 168.3 | 151.7 | +16.7 | 56.4% |
| **5** | **Nicholas Pooran** | **+0.286** | 823 | 525 | 156.8 | 150.3 | +6.4 | 55.6% |
| **6** | **Heinrich Klaasen** | **+0.279** | 739 | 437 | 169.1 | 161.5 | +7.6 | 55.4% |
| **7** | **David Miller** | **+0.273** | 707 | 496 | 142.5 | 139.1 | +3.4 | 56.9% |
| **8** | **Shimron Hetmyer** | **+0.271** | 599 | 373 | 160.6 | 148.0 | +12.6 | 52.6% |
| **9** | **Tristan Stubbs** | **+0.255** | 495 | 321 | 154.2 | 149.2 | +5.0 | 54.5% |
| **10** | **Liam Livingstone** | **+0.252** | 425 | 277 | 153.4 | 146.1 | +7.4 | 53.7% |

---

### Top 10 Bowlers Under Pressure
| Rank | Player | Pressure Score | Pressure Dots | Pressure Wkts | Pressure Balls | Pressure Econ | Baseline Econ | Econ Delta | Consistency |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Sunil Narine** | **+0.196** | 225 | 40 | 509 | 6.47 | 6.54 | -0.07 | 67.2% |
| **2** | **Jasprit Bumrah** | **+0.170** | 194 | 36 | 450 | 6.32 | 6.43 | -0.11 | 63.8% |
| **3** | **Matheesha Pathirana** | **+0.165** | 157 | 45 | 387 | 7.64 | 7.82 | -0.18 | 58.7% |
| **4** | **Harshal Patel** | **+0.163** | 179 | 53 | 473 | 8.84 | 9.08 | -0.24 | 56.4% |
| **5** | **Harpreet Brar** | **+0.136** | 129 | 24 | 311 | 7.14 | 7.29 | -0.15 | 60.0% |
| **6** | **Rashid Khan** | **+0.134** | 224 | 42 | 539 | 7.43 | 7.39 | +0.04 | 57.7% |
| **7** | **Arshdeep Singh** | **+0.132** | 220 | 55 | 560 | 8.65 | 8.69 | -0.04 | 54.5% |
| **8** | **Varun Chakravarthy** | **+0.129** | 187 | 37 | 451 | 7.50 | 7.48 | +0.02 | 57.1% |
| **9** | **Trent Boult** | **+0.128** | 184 | 32 | 412 | 7.74 | 7.91 | -0.17 | 56.5% |
| **10** | **Axar Patel** | **+0.125** | 178 | 32 | 428 | 7.15 | 7.18 | -0.03 | 58.3% |

---

## 🔬 Methodology Summary

```
Match Tier Weight (0.6 - 1.4)
           ×
Opponent Elo Multiplier (0.85 - 1.15)
           ×
Situational Tension (Win Probability Closeness + Required Rate / Over Leverage)
           =
     Delivery Pressure Index (PI)
```

1. **Stakes Tiering:**
   - Bilateral / Qualifiers: `0.60`
   - Tri-series: `0.70`
   - Emerging leagues / Continental group games: `0.60 – 0.70`
   - IPL League: `1.00`
   - Continental Finals: `0.85`
   - IPL Playoffs / Final: `1.40`
2. **Dynamic Team Elo:**
   - Independent rating pools for IPL franchises and international teams.
   - Evaluates opponent quality dynamically on the match date.
3. **Delivery-Level Pressure Modeling:**
   - Calibrated logistic win-probability model accounting for target, wickets down, balls remaining, and Elo advantage.
   - Computes state leverage and assigns each delivery a continuous Pressure Index.
4. **Performance Scoring & Shrinkage:**
   - Isolates the top 30% highest-pressure deliveries.
   - Benchmarks player returns against an 85% situation par + 15% player baseline hybrid.
   - Applies empirical Bayes shrinkage for low sample sizes and rewards consistency across pressure innings.

---

## 📁 Repository Structure

```
├── Dataset/
│   └── Data Pack/                    # Raw ball-by-ball match data (IPL & T20I 2022-2026)
├── outputs/
│   ├── METHODOLOGY.md                # Full technical report (Markdown)
│   ├── METHODOLOGY.html              # Standalone styled HTML report
│   ├── METHODOLOGY.pdf               # Publication-ready PDF documentation
│   ├── ball_pressure.csv             # Ball-by-ball calculated pressure indices
│   ├── player_pressure_ratings_bat.csv   # Complete batter pressure metrics
│   ├── player_pressure_ratings_bowl.csv  # Complete bowler pressure metrics
│   ├── top10_batters.csv             # Top 10 ranked batters
│   ├── top10_bowlers.csv             # Top 10 ranked bowlers
│   ├── summary.json                  # Overall dataset & metric summaries
│   └── figures/                      # Visualizations and metric distributions
├── pipeline.py                       # Main pipeline computing pressure indices and rankings
├── make_report.py                    # Generates plots, HTML, and compiled PDF report
├── sensitivity.py                    # Sensitivity analysis across weights & thresholds
└── README.md                         # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Recommended packages:
  ```bash
  pip install numpy matplotlib markdown beautifulsoup4 weasyprint
  ```
  *(Note: `pipeline.py` only requires `numpy` alongside standard library modules).*

### Execution

1. **Run the Rating Pipeline:**
   ```bash
   python pipeline.py
   ```
   *Generates all player metrics, pressure indices, and tabular outputs in `outputs/`.*

2. **Run Sensitivity Analysis:**
   ```bash
   python sensitivity.py
   ```

3. **Generate Full Report & Figures:**
   ```bash
   python make_report.py
   ```
   *Creates the figures in `outputs/figures/` and compiles `METHODOLOGY.html` & `METHODOLOGY.pdf`.*

---

## 📄 License & Attribution

Developed for **SupeRR Selector Season 2 — Round 1**. Match data covering 883 matches and 200,961 deliveries across IPL 2022–2026 and T20Is 2022–2026.
