# One Week Forecast of Electricity Demand in California

![cover image](images/cover.png)

[Live Dashboard](https://pipegalera.github.io/energy_forecasting/)

This dashboard shows the latest data on electricity demand for the main 4 primary electric utility companies in California:

- Pacific Gas and Electric (PGAE)
- Southern California Edison (SCE)
- San Diego Gas and Electric (SDGE)
- Valley Electric Association (VEA)

The `data` (US hourly demand for electricity) comes from EIA API. The predictions are made with `XGBoost` trained via `Optuna` for hypertunning. I tracked experiments and select the best models via `MLflow`.
The visualization is made with `plotly` package.

The data, forecasting, and visualization is refreshed daily using a `Docker` image run via `Github Actions` and deployed in a `Github page`.


## TODO

- [ ] Include MAPE metric within the visualization.
- [ ] Prediction Interval based on bootstrapping (https://otexts.com/fpp2/bootstrap.html) .
- [ ] Unit testing.
