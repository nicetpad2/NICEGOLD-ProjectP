
# Fixed Target Variable Creation
def create_improved_target(df, method = "multi_horizon_return"):
    """
    สร้าง target variable ที่มี predictive power สูงกว่า
    """
    if method == "multi_horizon_return":
        # ใช้ multiple horizon returns
        returns_1 = df["Close"].pct_change(1).shift( - 1)  # 1 - bar ahead
        returns_3 = df["Close"].pct_change(3).shift( - 3)  # 3 - bar ahead
        returns_5 = df["Close"].pct_change(5).shift( - 5)  # 5 - bar ahead

        # Weighted combination
        combined_return = (0.5 * returns_1 + 0.3 * returns_3 + 0.2 * returns_5)

        # Dynamic threshold based on volatility
        volatility = df["Close"].pct_change().rolling(20).std()
        threshold = volatility * 0.5  # Half of volatility as threshold

        target = (combined_return > threshold).astype(int)

    elif method == "volatility_adjusted":
        # Volatility - adjusted returns
        returns = df["Close"].pct_change().shift( - 5)
        volatility = returns.rolling(20).std()
        adjusted_returns = returns / (volatility + 1e - 8)

        # Use percentile - based threshold
        target = (adjusted_returns > adjusted_returns.quantile(0.6)).astype(int)

    elif method == "regime_aware":
        # Market regime - aware target
        returns = df["Close"].pct_change().shift( - 3)

        # Simple volatility regime
        vol = df["Close"].pct_change().rolling(10).std()
        vol_regime = vol > vol.rolling(50).median()

        # Different thresholds for different regimes
        threshold_high_vol = returns.quantile(0.55)
        threshold_low_vol = returns.quantile(0.65)

        target = np.where(
            vol_regime, 
            (returns > threshold_high_vol).astype(int), 
            (returns > threshold_low_vol).astype(int)
        )

    return pd.Series(target, index = df.index).fillna(0)

# Usage in your pipeline:
# target = create_improved_target(df, method = "multi_horizon_return")