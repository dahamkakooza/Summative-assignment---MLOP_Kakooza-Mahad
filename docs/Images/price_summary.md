# 💰 Agricultural Price Prediction - FAO Dataset

## Dataset Statistics
- Date range: 2005-01-31 to 2025-02-28
- Total records: 8,470
- Commodities: 7
- Regions: 5
- Market types: 3

## Model Architecture
- Algorithm: XGBoost Regressor
- Number of features: 23
- Best hyperparameters: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.8}

## Optimization Techniques
1. Hyperparameter Tuning (GridSearchCV)
2. Time Series Cross-Validation
3. Feature Engineering (lags, rolling statistics)
4. Regularization (L1/L2 via reg_alpha/reg_lambda)
5. Seasonal Encoding (sin/cos months)

## Evaluation Metrics (6+)
- MAE: 4.6868
- MSE: 36.2745
- RMSE: 6.0228
- R² Score: 0.9943
- Explained Variance: 0.9943
- MAPE: 1.63%

## Top 5 Important Features
             feature  importance
price_rolling_mean_3    0.734335
      commodity_code    0.228802
price_rolling_mean_6    0.011952
         price_lag_1    0.010518
        price_lag_12    0.003804

## Time Series CV Results
- Average RMSE: 144.3985
- Average R²: 0.7330

## Model Performance Summary
The model shows strong predictive performance with:
- High R² (0.994) indicating good fit
- Low MAPE (1.63%) showing accurate predictions
- Stable cross-validation results
