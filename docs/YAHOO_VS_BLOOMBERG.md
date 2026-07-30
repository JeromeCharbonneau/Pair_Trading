# Yahoo vs Bloomberg — Corn / Wheat comparison

Generated: 2026-07-29T23:43:26.926605+00:00

## Full-sample clean Corn/Wheat

| Metric | Yahoo | Bloomberg |
|---|---:|---:|
| Window | 2016-07-29 → 2026-07-29 | 2000-01-03 → 2026-07-29 |
| Clean N | 2,510 | 6,692 |
| Dropped rows | 4 | 144 |
| Return corr | 0.5286 | 0.5886 |
| Hedge ratio (log Corn on Wheat) | 0.9557 | 1.0304 |
| EG t-stat | -4.2135 | -5.1109 |
| EG p-value | 0.0035 | 0.0001 |
| Cointegrated at 5%? | True | True |
| Extreme \|r\|>10% flags | 7 | 9 |

## Overlap window (fairer head-to-head)

Common clean dates: **2,510** (2016-07-29 → 2026-07-29)

| Metric | Yahoo | Bloomberg |
|---|---:|---:|
| Return corr | 0.5286 | 0.5290 |
| Hedge ratio | 0.9557 | 0.9556 |
| EG t-stat | -4.2135 | -4.1507 |
| EG p-value | 0.0035 | 0.0043 |
| Cointegrated at 5%? | True | True |

## Exploratory screen rank of Corn / Wheat

- Yahoo rank: **2**
- Bloomberg rank: **3**

### Bloomberg top 5 pairs
             Pair  P-value  T-stat   Corr    N
 Wheat / Soybeans   0.0000 -5.3492 0.3816 6692
     Wheat / Rice   0.0001 -5.1475 0.1671 6689
     Corn / Wheat   0.0001 -5.1109 0.5851 6692
     Wheat / Oats   0.0005 -4.7206 0.3282 6693
Wheat / Crude Oil   0.0010 -4.5587 0.0783 6677

### Yahoo top 5 pairs
                   Pair  P-value  T-stat   Corr    N
           Oats / Wheat   0.0017 -4.4134 0.2845 2512
           Corn / Wheat   0.0035 -4.2135 0.5287 2510
      Crude Oil / Wheat   0.0107 -3.8766 0.0356 2511
Crude Oil / Natural Gas   0.0109 -3.8695 0.0075 2512
        Corn / Soybeans   0.0129 -3.8159 0.5400 2510

## Interpretation notes
- Bloomberg history is much longer (~2000+); full-sample EG results are **not** directly comparable to Yahoo’s ~10y window without the overlap table.
- Cleaning rules identical on both sources.
- Bloomberg sheets are still continuous/vendor series; roll methodology remains a disclosed limitation until true per-contract rolls are built.
- **Primary research series for Days 3–4:** Bloomberg. Yahoo is retained only as a provisional comparison / Day-1 screen.
