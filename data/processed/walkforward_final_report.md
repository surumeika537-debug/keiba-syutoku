# Walk-Forward Validation Final Report — F_D9_DYNAMIC_STATE
_generated: walkforward_validation.py / 6 folds / audit pass = True_
## Leakage Audit
- all checks PASS: **True**
- folds audited: 6
  - fold 1 (2015-2019 → 2020): PASS
  - fold 2 (2015-2020 → 2021): PASS
  - fold 3 (2015-2021 → 2022): PASS
  - fold 4 (2015-2022 → 2023): PASS
  - fold 5 (2015-2023 → 2024): PASS
  - fold 6 (2015-2024 → 2025): PASS

## F per fold (test year out-of-sample)
| fold | year | races | invest | profit | ROI | DD | hits | skipped | halt | skipped_winners |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2020 | 20 | ¥7,740 | ¥+95,238 | +1230.5% | 2.2% | 2 | 13 | 13 | 1 |
| 2 | 2021 | 22 | ¥8,750 | ¥+11,238 | +128.4% | 5.0% | 2 | 9 | 9 | 0 |
| 3 | 2022 | 23 | ¥8,250 | ¥-1,619 | -19.6% | 6.4% | 1 | 8 | 8 | 0 |
| 4 | 2023 | 17 | ¥16,495 | ¥-16,495 | -100.0% | 16.5% | 0 | 0 | 0 | 0 |
| 5 | 2024 | 15 | ¥8,250 | ¥+42,388 | +513.8% | 2.7% | 2 | 0 | 0 | 0 |
| 6 | 2025 | 21 | ¥4,500 | ¥+19,716 | +438.1% | 2.1% | 1 | 14 | 14 | 1 |

**F aggregate (6 OOS years)**: profit ¥+150,465, E baseline: ¥+88,130, F-E gap: ¥+62,335

## 2023 fold survival analysis
- F profit: ¥-16,495  (E: ¥-9,200)
- HALT events: 0
- RED events: 0
- CHAOTIC regime races: 2
- DARK_SUPPRESSED regime races: 0
- skipped: 0  / skipped_winners: 0
- max DD: 16.5%
- **catastrophic DD prevented: False**

## Parameter drift across folds
| threshold | min | max | range/mean |
|---|---:|---:|---:|
| regime_fav_low | 0.200 | 0.233 | 15.8% |
| regime_fav_high | 0.400 | 0.433 | 8.0% |
| regime_dark_low | 0.733 | 0.767 | 4.4% |
| payout_inflation_threshold | 1.561 | 1.799 | 14.1% |
| cv_low_threshold | 1.341 | 1.460 | 8.5% |

## Monte Carlo (within-year shuffle + ±10% payout perturbation)
- trials: 1000
- ruin rate: 0.0%
- median return: ¥+204,030
- p05 return: ¥+187,028
- p95 return: ¥+221,090
- p95 drawdown: 18.3%
- p99 drawdown: 18.4%
- worst fold mode (most stressful year): 2023

## Final Verdict
### **STATISTICALLY ROBUST**
F が E を上回り (+150465 vs +88130), 4/6 fold で profit > 0, 平均 DD 5.8% < 20%.

### Should F advance to live paper trading?
**YES** — promote to weekly paper trading with current config.
