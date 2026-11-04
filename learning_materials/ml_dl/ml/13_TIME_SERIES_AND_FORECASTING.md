# ML13. Time-series forecasting

Forecasting predicts a future value using information available when the prediction is made. A time series is a sequence of observations ordered in time, such as daily demand. A forecast origin is the prediction time; the horizon is how far ahead we predict. These boundaries determine which inputs are legitimate.

Our observed history is [3, 4, 5]. We predict the next value using the average of the most recent two or three values. The actual next value, 6, is kept outside that calculation and used only to score the forecast.

Ordinary random row splits can be misleading when future observations help predict the past. Forecasting instead tests a procedure across appropriate time cutoffs.

[course map](../COURSE_MAP.md). Required background: ML03, ML10.

## 1. Time order, lags and seasonality

**Key terms.** A **time series** is ordered by time; a **lag** is a past observation; a **horizon** is how far ahead the forecast targets; **seasonality** repeats a pattern. **Forecast origin** is the latest information time.

**Process.** At each origin, construct features using only available history, fit on earlier data and predict the chosen horizon. Advance the origin and repeat for development checks; reserve a final later period for frozen evaluation.

A random split can let neighbouring or future observations inform training for a past test case. Lag construction and rolling summaries must respect their information timestamps. For horizon h, a feature known only at target time is not available at forecast origin unless a separate forecast supplies it.

A last-value or seasonal-naive predictor is a serious baseline. Trend/seasonality decomposition, exponential smoothing and ARIMA are alternative statistical families; a linear/boosted model on lagged features is another. Neural sequence models should be compared with these baselines, not only with an uninformative constant. Nonstationarity can make historical relationships fail.

## 2. A worked calculation

**Symbols and assumptions.** y_t is the observation at time t; h is forecast horizon; s is seasonal period. A seasonal-naive forecast repeats the latest available corresponding seasonal value. Error is prediction minus future observation.

For an hourly series with a daily period s=24, a one-hour-ahead forecast at origin t may use y_(t+1−24)=y_(t−23). If this historical hour has 120 rentals and the later observed target is 150, absolute error is 30. A feature containing observed y_(t+1) is leakage, regardless of model type.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Rolling windows and forecast horizons

Expanding windows keep all past history; rolling windows discard old observations. A temporal gap can prevent overlap between constructed inputs/labels across boundaries, but its length must follow the feature/horizon design. Ordinary iid confidence intervals are not automatically valid for correlated forecast errors.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: prevent future-information leakage

Use Bike Sharing with a stated demand-prediction setting. Exclude casual and registered because they sum to cnt. Distinguish known calendar information from target-time measured weather. Do not mix hourly and daily records as independent counts of different activity.

**Check.** Why can an LSTM with excellent shuffled-hour accuracy still fail as tomorrow's forecaster?

**Answer.** Its evaluation may expose future conditions or related observations and may not match the forecast horizon. Rebuild chronological, availability-aware comparisons before changing architecture.

## Sources

- [Forecasting: Principles and Practice, third edition](https://otexts.com/fpp3/).
- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
