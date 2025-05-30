#!/usr/bin/env python3
"""
Technical Indicators - Provides technical analysis indicators for financial data

Implements common technical indicators used in financial analysis:
- Moving averages (SMA, EMA, WMA)
- Oscillators (RSI, MACD, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, Money Flow Index)
- Trend indicators (ADX, Ichimoku Cloud)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from loguru import logger


class TechnicalAnalyzer:
    """
    Analyzes technical indicators and generates trading signals.
    
    Provides comprehensive technical analysis tools and scoring mechanisms
    for trading strategies.
    """
    
    def __init__(self):
        """
        Initialize the technical analyzer.
        """
        logger.info("Technical analyzer initialized")
        self.default_periods = {
            'sma': 20,
            'ema': 20,
            'rsi': 14,
            'macd': (12, 26, 9),
            'stochastic': (14, 3, 3),
            'bollinger': (20, 2),
            'adx': 14
        }

    def calculate_sma(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            data: Input price series
            period: Lookback period
            
        Returns:
            SMA series
        """
        return data.rolling(window=period).mean()

    def calculate_ema(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Input price series
            period: Lookback period
            
        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Relative Strength Index
        
        Args:
            data: Input price DataFrame
            period: Lookback period
            
        Returns:
            Latest RSI value
        """
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 0.0

    def calculate_macd(self, data: pd.Series, 
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> Tuple[float, float]:
        """
        Calculate MACD and Signal Line
        
        Args:
            data: Input price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (MACD, Signal)
        """
        try:
            fast_ema = self.calculate_ema(data, fast_period)
            slow_ema = self.calculate_ema(data, slow_period)
            macd = fast_ema - slow_ema
            signal = self.calculate_ema(macd, signal_period)
            return macd.iloc[-1], signal.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0, 0.0

    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate trend strength score
        
        Args:
            data: Input price DataFrame
            
        Returns:
            Trend strength score (0-100)
        """
        try:
            # Calculate multiple moving averages
            sma20 = self.calculate_sma(data['close'], 20)
            sma50 = self.calculate_sma(data['close'], 50)
            sma100 = self.calculate_sma(data['close'], 100)
            
            # Calculate trend direction scores
            short_term = (data['close'].iloc[-1] > sma20.iloc[-1]) * 100
            medium_term = (data['close'].iloc[-1] > sma50.iloc[-1]) * 100
            long_term = (data['close'].iloc[-1] > sma100.iloc[-1]) * 100
            
            # Calculate momentum score
            momentum = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20] * 100
            
            # Combine scores
            trend_score = (short_term + medium_term + long_term + momentum) / 4
            return min(max(trend_score, 0), 100)
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0

    def score_company(self, data: Dict[str, List[float]]) -> float:
        """
        Calculate overall technical score for a company
        
        Args:
            data: Dictionary containing price history
            
        Returns:
            Overall technical score (0-100)
        """
        try:
            df = pd.DataFrame(data)
            
            # Calculate indicators
            rsi = self.calculate_rsi(df)
            macd, signal = self.calculate_macd(df['close'])
            trend_strength = self.calculate_trend_strength(df)
            
            # Calculate scores
            rsi_score = 100 - abs(rsi - 50)  # Close to 50 is neutral
            macd_score = 100 if macd > signal else 0  # MACD above signal is bullish
            
            # Combine scores
            total_score = (rsi_score + trend_strength + macd_score) / 3
            return min(max(total_score, 0), 100)
        except Exception as e:
            logger.error(f"Error scoring company: {e}")
            return 0.0
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all major technical indicators to a price dataframe.
        
        Args:
            df: DataFrame with OHLCV data (must have columns: Open, High, Low, Close, Volume)
            
        Returns:
            DataFrame with added technical indicators
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in result.columns for col in required_columns):
            logger.warning("DataFrame missing required OHLCV columns")
            return result
        
        # Add moving averages
        result = self.add_moving_averages(result)
        
        # Add oscillators
        result = self.add_oscillators(result)
        
        # Add volatility indicators
        result = self.add_volatility_indicators(result)
        
        # Add volume indicators
        result = self.add_volume_indicators(result)
        
        # Add trend indicators
        result = self.add_trend_indicators(result)
        
        return result
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add moving average indicators to the dataframe.
        
        Args:
            df: DataFrame with at least Close prices
            
        Returns:
            DataFrame with added moving average indicators
        """
        result = df.copy()
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            result[f'SMA_{period}'] = self.sma(result['Close'], period)
        
        # Exponential Moving Averages
        for period in [5, 12, 26, 50, 200]:
            result[f'EMA_{period}'] = self.ema(result['Close'], period)
        
        return result
    
    def add_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add oscillator indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added oscillator indicators
        """
        result = df.copy()
        
        # RSI
        result['RSI_14'] = self.rsi(result['Close'], 14)
        
        # MACD
        macd_line, signal_line, histogram = self.macd(result['Close'])
        result['MACD_Line'] = macd_line
        result['MACD_Signal'] = signal_line
        result['MACD_Histogram'] = histogram
        
        # Stochastic Oscillator
        result['Stoch_K'], result['Stoch_D'] = self.stochastic_oscillator(result)
        
        return result
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added volatility indicators
        """
        result = df.copy()
        
        # Bollinger Bands
        result['BB_Upper'], result['BB_Middle'], result['BB_Lower'] = self.bollinger_bands(result['Close'])
        
        # Average True Range
        result['ATR_14'] = self.atr(result, 14)
        
        return result
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added volume indicators
        """
        result = df.copy()
        
        # On-Balance Volume
        result['OBV'] = self.obv(result)
        
        # Money Flow Index
        result['MFI_14'] = self.money_flow_index(result, 14)
        
        # Volume Moving Average
        result['Volume_SMA_20'] = self.sma(result['Volume'], 20)
        
        return result
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added trend indicators
        """
        result = df.copy()
        
        # Average Directional Index
        result['ADX_14'] = self.adx(result, 14)
        
        # Ichimoku Cloud (simplified)
        tenkan, kijun, senkou_a, senkou_b, chikou = self.ichimoku_cloud(result)
        result['Ichimoku_Tenkan'] = tenkan
        result['Ichimoku_Kijun'] = kijun
        result['Ichimoku_Senkou_A'] = senkou_a
        result['Ichimoku_Senkou_B'] = senkou_b
        result['Ichimoku_Chikou'] = chikou
        
        return result
    
    # ===== MOVING AVERAGES =====
    
    def sma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            series: Price series
            period: Moving average period
            
        Returns:
            Series containing the SMA values
        """
        return series.rolling(window=period).mean()
    
    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            series: Price series
            period: Moving average period
            
        Returns:
            Series containing the EMA values
        """
        return series.ewm(span=period, adjust=False).mean()
    
    def wma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Weighted Moving Average.
        
        Args:
            series: Price series
            period: Moving average period
            
        Returns:
            Series containing the WMA values
        """
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
    
    # ===== OSCILLATORS =====
    
    def rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            series: Price series
            period: RSI period
            
        Returns:
            Series containing the RSI values
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(self, series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence.
        
        Args:
            series: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        # Calculate MACD line
        fast_ema = self.ema(series, fast_period)
        slow_ema = self.ema(series, slow_period)
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = self.ema(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def stochastic_oscillator(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            df: DataFrame with High, Low, Close columns
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D)
        """
        # Calculate %K
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        
        # Calculate %D
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    # ===== VOLATILITY INDICATORS =====
    
    def bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            series: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        middle_band = self.sma(series, period)
        std = series.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with High, Low, Close columns
            period: ATR period
            
        Returns:
            Series containing the ATR values
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    # ===== VOLUME INDICATORS =====
    
    def obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume.
        
        Args:
            df: DataFrame with Close and Volume columns
            
        Returns:
            Series containing the OBV values
        """
        close = df['Close']
        volume = df['Volume']
        
        # Calculate price direction
        direction = np.sign(close.diff())
        direction = direction.fillna(0).astype(int)
        
        # Calculate OBV
        obv = (direction * volume).cumsum()
        
        return obv
    
    def money_flow_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index.
        
        Args:
            df: DataFrame with OHLCV data
            period: MFI period
            
        Returns:
            Series containing the MFI values
        """
        # Calculate typical price
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * df['Volume']
        
        # Get positive and negative money flow
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        
        # Calculate money flow ratio
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        money_flow_ratio = positive_sum / negative_sum
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    # ===== TREND INDICATORS =====
    
    def adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index.
        
        Args:
            df: DataFrame with OHLC data
            period: ADX period
            
        Returns:
            Series containing the ADX values
        """
        # Calculate +DM and -DM
        high_diff = df['High'].diff()
        low_diff = df['Low'].diff().multiply(-1)
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR
        atr_series = self.atr(df, period)
        
        # Calculate +DI and -DI
        plus_di = 100 * (self.ema(plus_dm, period) / atr_series)
        minus_di = 100 * (self.ema(minus_dm, period) / atr_series)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = self.ema(dx, period)
        
        return adx
    
    def ichimoku_cloud(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate Ichimoku Cloud components.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Tuple of (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price shifted back 26 periods
        chikou_span = close.shift(-26)
        
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span