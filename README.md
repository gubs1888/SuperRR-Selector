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
| **1** | **Shreyas Iyer** | **+0.333** | 1192 | 696 | 171.3 | 153.1 | +18.1 | 51.1% |
| **2** | **Suryakumar Yadav** | **+0.330** | 1505 | 881 | 170.8 | 166.3 | +4.5 | 62.7% |
| **3** | **Shubman Gill** | **+0.329** | 1295 | 756 | 171.3 | 149.4 | +21.9 | 43.5% |
| **4** | **Rajat Patidar** | **+0.318** | 992 | 557 | 178.1 | 167.9 | +10.2 | 53.3% |
| **5** | **Nicholas Pooran** | **+0.286** | 1351 | 815 | 165.8 | 157.1 | +8.7 | 53.4% |
| **6** | **Ruturaj Gaikwad** | **+0.278** | 768 | 464 | 165.5 | 139.0 | +26.5 | 48.6% |
| **7** | **Jos Buttler** | **+0.276** | 1263 | 781 | 161.7 | 155.4 | +6.3 | 50.0% |
| **8** | **Lokesh Rahul** | **+0.272** | 1262 | 781 | 161.6 | 141.3 | +20.3 | 48.6% |
| **9** | **Heinrich Klaasen** | **+0.224** | 1652 | 933 | 177.1 | 166.8 | +10.3 | 58.5% |
| **10** | **Ishan Kishan** | **+0.221** | 892 | 524 | 170.2 | 146.4 | +23.9 | 40.5% |

---

### Top 10 Bowlers Under Pressure
| Rank | Player | Pressure Score | Pressure Dots | Pressure Wkts | Pressure Balls | Pressure Econ | Baseline Econ | Econ Delta | Consistency |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Sunil Narine** | **+0.196** | 369 | 48 | 1006 | 6.71 | 6.87 | -0.16 | 62.1% |
| **2** | **Jasprit Bumrah** | **+0.170** | 238 | 36 | 677 | 7.55 | 7.27 | +0.28 | 50.9% |
| **3** | **Matheesha Pathirana** | **+0.165** | 203 | 41 | 573 | 7.93 | 8.85 | -0.92 | 53.8% |
| **4** | **Harshal Patel** | **+0.163** | 306 | 59 | 896 | 8.10 | 9.32 | -1.22 | 52.5% |
| **5** | **Harpreet Brar** | **+0.136** | 133 | 22 | 424 | 7.87 | 8.25 | -0.38 | 48.4% |
| **6** | **Noor Ahmad** | **+0.128** | 266 | 44 | 784 | 7.93 | 8.01 | -0.08 | 54.3% |
| **7** | **Wanindu Hasaranga** | **+0.127** | 234 | 43 | 688 | 8.31 | 7.94 | +0.37 | 47.5% |
| **8** | **Krunal Pandya** | **+0.090** | 267 | 42 | 879 | 7.49 | 7.81 | -0.32 | 46.3% |
| **9** | **Kuldeep Yadav** | **+0.069** | 361 | 64 | 1135 | 7.89 | 8.02 | -0.13 | 62.3% |
| **10** | **Varun Chakravarthy** | **+0.041** | 362 | 48 | 972 | 7.83 | 8.01 | -0.18 | 55.1% |

*Pressure dots count legal pressure balls conceding no runs. Pressure score is the final ranking
score from `outputs/top10_bowlers.csv`; all columns are reproduced from the pipeline outputs.*

---

## 🔬 Methodology Summary

```
Match Tier Weight (0.6 - 1.4)
           ×
Opponent Elo Multiplier (0.80 - 1.20)
           ×
Situational Tension (Win Probability Closeness + Required Rate / Over Leverage)
           =
     Delivery Pressure Index (PI)
```

1. **Stakes Tiering:**
   - Bilateral / Qualifiers: `0.60`
   - Emerging leagues: `0.60`
   - Tri-series: `0.70`
   - Asia Cup league (group / Super 4): `0.70`
   - Emerging knockouts: `0.75`
   - Asia Cup final: `0.85`
   - IPL League: `1.00`
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
  pip install numpy matplotlib markdown beautifulsoup4
  ```
  *(Note: `pipeline.py` only requires `numpy` alongside standard library modules. PDF rendering uses headless Google Chrome/Chromium, with LibreOffice as a fallback.)*

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
