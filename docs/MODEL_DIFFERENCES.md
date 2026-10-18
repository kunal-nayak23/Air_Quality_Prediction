# Regression models versus the AQI bucket classifier

All four benchmark models solve **regression**: they estimate a continuous numerical AQI value from the same 11 pollutant measurements. They share the same labeled station-day records, date-grouped 80/20 holdout, and training-only median imputation.

| Model | Core idea | Scaling | Main strength | Main limitation | Verified test RMSE |
|---|---|---|---|---|---:|
| KNN Regressor | Average the AQI values of nearby training examples, weighted by distance | Required | Flexible local patterns; intuitive | Slow prediction and sensitive to distance/noise | 42.1357 |
| SVR with RBF kernel | Fit a smooth nonlinear function while ignoring errors inside an epsilon tube | Required | Strong nonlinear fit and controlled complexity | Expensive on large datasets; less interpretable | 42.2024 |
| Random Forest Regressor | Average many randomized decision trees | Not required | Robust nonlinear interactions; strong baseline; feature importance | Larger artifact; piecewise predictions | **38.8910** |
| XGBoost Regressor | Add shallow trees sequentially to correct residual errors | Not required | Efficient boosting and strong tabular performance | More tuning-sensitive; sequential learning can overfit | 39.2731 |

The deployed numerical model remains the existing **Random Forest Regressor**, because it has the lowest verified RMSE in this benchmark. None of the numbered notebooks replaces that file.

The separate **XGBoost AQI bucket classifier** does not predict a number. It chooses one of six labels (`Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, `Severe`) and can provide class probabilities. It was evaluated with classification measures (accuracy 0.794006 and macro-F1 0.779885), not MAE/RMSE/R². The classifier and numerical regressor are independent estimates, so their displayed bucket and number-derived bucket can occasionally disagree.

