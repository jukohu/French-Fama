
## Fama–French 3-Factor Model (Alphabet case study)

This mini-project practices Python workflow on a topic that’s highly relevant to **quantitative investing**.

The **Fama–French (1993) 3-factor model** extends CAPM by explaining stock (excess) returns with three systematic factors:

1. **Market excess return** $(R_m - R_f)$
2. **SMB** (*Small Minus Big*): small-cap minus large-cap outperformance
3. **HML** (*High Minus Low*): value (high book/market) minus growth (low book/market) outperformance

Model (total return form):

$$
r = R_f + \beta\,(R_m - R_f) + b_s\,\mathrm{SMB} + b_v\,\mathrm{HML} + \alpha
$$

In many datasets, these factors explain substantially more variation in returns than CAPM.

In this project, I use Fama–French factors to **explain and (simply) predict** Alphabet’s monthly returns. Parts of the structure are inspired by *fischerleben’s* “Algorithmic Trading Project”.

---

## Setup

```python
# Initial imports
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Modeling
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Visualization
import matplotlib.pyplot as plt
# %matplotlib inline  # uncomment in a notebook
```

---

## Load Fama–French monthly factors

Ken French CSVs include preamble/footer text, so we extract the actual monthly block safely.

```python
import re, io

def read_kf_monthly(path):
    # Read raw lines (Ken French often uses latin-1)
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    # Find first monthly data row: YYYYMM,...
    start = next((i for i, ln in enumerate(lines) if re.match(r"^\s*\d{6}\s*,", ln)), None)
    if start is None:
        raise ValueError("Could not find a YYYYMM data row.")
    header = start - 1

    # Find end of the monthly block (blank/new section)
    end = None
    for j in range(start, len(lines)):
        if lines[j].strip() == "" or re.search(r"\b(Annual|Weekly|Daily)\b", lines[j]):
            end = j
            break
    end = len(lines) if end is None else end

    csv_block = "".join([lines[header]] + lines[start:end])
    df = pd.read_csv(io.StringIO(csv_block), engine="python")
    df.columns = [c.strip() for c in df.columns]

    # Normalize date and index
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
    df = df.set_index("Date").sort_index()

    return df

french = read_kf_monthly("F-F_Research_Data_Factors.csv")
french.head()
```

> Note: Fama–French columns are usually in **percent**; we’ll convert to decimals later.

---

## Load Alphabet monthly prices (EUR) and merge

```python
# Load your Excel (sheet with monthly data)
alphabet = pd.read_excel("Alphabet Inc._StockChart_08_29_2025.xlsx", sheet_name="Data")

# Clean & align to monthly PeriodIndex
ff = french.copy()
ff.index = ff.index.to_period("M")  # YYYY-MM

alphabet_clean = (
    alphabet.rename(columns={
        "Pricing Date": "Date",
        "GOOGL | Share Price (Monthly)(€)": "GOOGL_Price_EUR",
        "GOOGL | Volume (Monthly)": "GOOGL_Volume"
    })
    .assign(Date=lambda d: pd.to_datetime(d["Date"]))
    .set_index("Date")
    .sort_index()
)
alphabet_clean.index = alphabet_clean.index.to_period("M")
alphabet_clean = alphabet_clean[~alphabet_clean.index.duplicated(keep="last")]

# Merge (inner) on monthly period
combined_df = ff.join(alphabet_clean[["GOOGL_Price_EUR"]], how="inner")

# Convert FF factors to decimals if they look like percents
for col in ["Mkt-RF","SMB","HML","RF"]:
    if col in combined_df.columns and combined_df[col].abs().max() > 1:
        combined_df[col] = combined_df[col] / 100.0

combined_df.head()
```

---

## Compute Alphabet monthly returns

```python
combined_df = combined_df.sort_index()
combined_df["GOOGL_Ret"] = combined_df["GOOGL_Price_EUR"].astype(float).pct_change()
combined_df = combined_df.dropna(subset=["GOOGL_Ret"])
combined_df.head()
```

---

## Train / test split

```python
X = combined_df.drop(columns=["GOOGL_Ret","GOOGL_Price_EUR"])
y = combined_df["GOOGL_Ret"]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
price_test = combined_df["GOOGL_Price_EUR"][split:]
```

---

## OLS (scikit-learn) and predictions

```python
lin_reg_model = LinearRegression(fit_intercept=True)
lin_reg_model.fit(X_train, y_train)
predictions = pd.Series(lin_reg_model.predict(X_test), index=X_test.index, name="Pred")
```

---

## Signals (no look-ahead; long-only)

A safer rule converts the forecast to a **binary position** using **only information at time *t*** and **trades at *t+1***.
Here: go long if predicted **excess** return > 0 (or > small cost).

```python
y_test_df = y_test.to_frame(name="GOOGL_Ret").copy()
y_test_df["GOOGL_Price_EUR"] = price_test
y_test_df = y_test_df.join(predictions)

# If RF exists, define predicted excess return; else just use Pred
pred_excess = y_test_df["Pred"] - (X_test["RF"] if "RF" in X_test.columns else 0.0)

# Raw signal at time t
y_test_df["Buy_Signal_raw"] = (pred_excess > 0.0).astype(float)

# Execute next period to avoid look-ahead
y_test_df["Buy_Signal"] = y_test_df["Buy_Signal_raw"].shift(1).fillna(0)
y_test_df.head()
```

---

## Backtest: position, P\&L, equity curve (monthly)

```python
def generate_signals(input_df, start_capital=100_000, share_count=2_000):
    df = input_df.copy().sort_index()

    share_size = share_count
    df["Position"] = share_size * df["Buy_Signal"]
    df["Entry/Exit"] = df["Buy_Signal"].diff().fillna(0.0)
    df["Entry/Exit Position"] = df["Position"].diff().fillna(0.0)

    # Holdings & cash
    df["Holdings"] = df["Entry/Exit Position"].cumsum() * df["GOOGL_Price_EUR"]
    df["Cash"] = start_capital - (df["GOOGL_Price_EUR"] * df["Entry/Exit Position"]).cumsum()

    df["Total"] = df["Cash"] + df["Holdings"]
    df["Period_Return"] = df["Total"].pct_change()
    df["Cumulative_Return"] = (1 + df["Period_Return"]).cumprod() - 1

    return df.dropna(subset=["Period_Return"])

signals_df = generate_signals(y_test_df)
signals_df.head(10)
```

---

## Evaluation (monthly → annualized with 12)

```python
def algo_evaluation(signals_df, periods_per_year=12):
    ann = periods_per_year
    out = pd.DataFrame(index=[
        "Annual Return","Cumulative Returns","Annual Volatility","Sharpe Ratio","Sortino Ratio"
    ], columns=["Backtest"])

    out.loc["Cumulative Returns"] = signals_df["Cumulative_Return"].iloc[-1]
    mean_r = signals_df["Period_Return"].mean()
    std_r  = signals_df["Period_Return"].std()

    out.loc["Annual Return"]     = mean_r * ann
    out.loc["Annual Volatility"] = std_r * np.sqrt(ann)
    out.loc["Sharpe Ratio"]      = (mean_r * ann) / (std_r * np.sqrt(ann)) if std_r>0 else np.nan

    # Sortino (downside std)
    dr = signals_df["Period_Return"].copy()
    downside = dr[dr < 0]**2
    down_stdev = np.sqrt(downside.mean()) * np.sqrt(ann) if len(downside)>0 else np.nan
    sortino = (mean_r * ann) / down_stdev if down_stdev and down_stdev>0 else np.nan
    out.loc["Sortino Ratio"] = sortino

    return out

algo_evaluation(signals_df)
```

### Buy-and-hold comparison

```python
def underlying_evaluation(signals_df, periods_per_year=12):
    u = pd.DataFrame(index=signals_df.index)
    u["GOOGL_Price_EUR"] = signals_df["GOOGL_Price_EUR"]
    u["Period_Return"] = u["GOOGL_Price_EUR"].pct_change().fillna(0.0)
    u["Cumulative_Return"] = (1 + u["Period_Return"]).cumprod() - 1
    return algo_evaluation(u.dropna(subset=["Period_Return"]), periods_per_year)

def algo_vs_underlying(signals_df, periods_per_year=12):
    cols = ["Algo","Underlying"]
    comp = pd.DataFrame(index=[
        "Annual Return","Cumulative Returns","Annual Volatility","Sharpe Ratio","Sortino Ratio"
    ], columns=cols)
    comp["Algo"] = algo_evaluation(signals_df, periods_per_year)["Backtest"]
    comp["Underlying"] = underlying_evaluation(signals_df, periods_per_year)["Backtest"]
    return comp

algo_vs_underlying(signals_df)
```

---

## StatsModels OLS (coefs, t-stats, $R^2$)

```python
y = combined_df["GOOGL_Ret"]
X = combined_df.drop(columns=["GOOGL_Ret","GOOGL_Price_EUR"])
X = sm.add_constant(X)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

ols = sm.OLS(y_train, X_train).fit()
print(ols.summary())

# Partial regression plots (on train fit)
fig = sm.graphics.plot_partregress_grid(ols, fig=plt.figure(figsize=(12,8)))
plt.show()
```

---

## Cumulative returns plot

```python
def underlying_returns(signals_df):
    df = pd.DataFrame(index=signals_df.index)
    df["Underlying_Return"] = signals_df["GOOGL_Price_EUR"].pct_change().fillna(0.0)
    df["Underlying_Cumulative"] = (1 + df["Underlying_Return"]).cumprod() - 1
    df["Algo_Cumulative"] = signals_df["Cumulative_Return"]
    return df[["Underlying_Cumulative","Algo_Cumulative"]]

underlying_returns(signals_df).plot(figsize=(14,6), title="Cumulative Returns: Algo vs Buy-and-Hold")
plt.show()
```

---

## Notes & limitations

* Data are **monthly**; annualization uses **12**.
* Signals are **shifted** one period to avoid look-ahead.
* **Transaction costs, slippage, taxes, borrow fees** are ignored here—add them before drawing conclusions.
* Fama–French factors are used as **predictors**; this is a **predictive** exercise, not a causal claim.
* Results on a single stock are noisy; consider cross-sectional tests and robustness checks (rolling windows, out-of-sample).

