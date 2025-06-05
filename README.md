# AI Hedge Fund

A comprehensive AI-driven hedge fund infrastructure featuring multiple legendary investor AI strategies, economic analysis bots, real-time market analysis, portfolio management, and local AI model training.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-HedgeFund.git
   cd AI-HedgeFund
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env` and update with your API keys
   - For development, you can use the default configuration which uses free data sources

### Running the Dashboard

#### Production Mode

1. Start the dashboard:
   ```bash
   python run_dashboard.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8088
   ```

3. The API server will be available at:
   ```
   http://localhost:8089
   ```

#### Development Mode

For development with auto-reload:

```bash
# Install development dependencies
pip install watchdog

# Run in development mode
python run_dev.py
```

The development server will:
- Automatically restart when Python files, YAML configs, or .env files change
- Open the dashboard in your default web browser
- Provide better error messages and logging

### Testing

Run the full test suite:

```bash
# Run all tests
python run_tests.py

# Run unit tests only
pytest tests/unit/

# Run functional tests only
pytest tests/functional/

# Run with coverage
pytest --cov=./ tests/
```

### Testing

To run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov requests

# Run tests
pytest tests/ -v

# Run with coverage report
pytest --cov=./ tests/ -v
```

## System Architecture

The AI Hedge Fund system is built with a modular architecture that separates concerns and allows for flexible deployment and scaling. Below is a high-level overview of the system architecture:

```
+---------------------+    +------------------------+    +----------------------+
|    DATA SOURCES     |    |    AGENT STRATEGIES    |    |   SYSTEM SERVICES   |
+---------------------+    +------------------------+    +----------------------+
| - Market Data       |    | - Warren Buffett      |    | - Dashboard (Dash) (8080)  |
| - Economic Data     |----| - Peter Lynch         |----| - API Server (8081) |
| - Fundamental Data  |    | - Benjamin Graham     |    | - Risk Management   |
| - Sentiment Data    |    | - Ray Dalio           |    | - Portfolio Mgmt    |
| - Technical Metrics |    | - George Soros        |    | - Backtesting       |
+---------------------+    | - Joel Greenblatt     |    | - Model Training    |
         |                 | - Economic Strategies  |    +----------------------+
         |                 +------------------------+              |
         |                            |                            |
         v                            v                            v
+-----------------------------------------------------------------------------------+
|                              CORE INFRASTRUCTURE                                  |
+-----------------------------------------------------------------------------------+
| +---------------+  +----------------+  +----------------+  +-------------------+  |
| | Data Manager  |->| Strategy Mgr  |->| Portfolio Mgr  |->| Risk Manager     |  |
| +---------------+  +----------------+  +----------------+  +-------------------+  |
|                                  ^                                                |
|                                  |                                                |
|                        +-----------------+                                        |
|                        | AI Model Manager|                                        |
|                        +-----------------+                                        |
+-----------------------------------------------------------------------------------+
```

## Core Components

### Data Manager

The Data Manager is responsible for collecting, processing, and providing access to various types of financial data:

- Market data (stocks, ETFs, indices)
- Economic indicators
- Fundamental data
- Sentiment data
- Technical indicators

It integrates with multiple data sources including Alpha Vantage, FRED, Finnhub, and Quandl, and implements caching to optimize performance.

### Strategy Manager

The Strategy Manager orchestrates multiple AI-driven investment strategies, including models based on legendary investors:

- Warren Buffett (Value Investing)
- Peter Lynch (Growth at Reasonable Price)
- Benjamin Graham (Deep Value)
- Ray Dalio (All Weather)
- George Soros (Reflexivity)
- Joel Greenblatt (Magic Formula)
- Economic and macro strategies

Each strategy generates signals that are aggregated to provide unified portfolio recommendations.

### Portfolio Manager

The Portfolio Manager handles all aspects of portfolio construction and management:

- Portfolio construction and optimization
- Position sizing and allocation
- Rebalancing and execution
- Performance tracking
- Transaction cost analysis

### Risk Manager

The Risk Manager monitors and controls risk exposure:

- Risk metrics calculation (VaR, CVaR, Sharpe ratio, etc.)
- Exposure limits enforcement
- Drawdown protection
- Correlation analysis

### AI Model Manager

The AI Model Manager handles the training, evaluation, and deployment of machine learning models:

- Model training and validation
- Hyperparameter optimization
- Model persistence and versioning
- Inference services

## UI Components

### Dashboard

A sleek, minimalist web-based dashboard for monitoring and controlling the system:

- Portfolio performance visualization and holdings management
- Strategy performance metrics and signal tracking
- Market data and economic indicators visualization
- Risk metrics and alerts monitoring
- System configuration and control

The dashboard is built with Dash and provides an intuitive and modern interface for users.

#### Launching the Dashboard

To launch the Dash dashboard, run the following command from the project root directory:

```bash
python run_dashboard.py
```

Once launched, the dashboard will be available at http://localhost:8080 by default.
### API Server

A RESTful API server that provides programmatic access to the system:

- Portfolio status and performance endpoints
- Strategy signals and performance metrics
- Market data and economic indicators
- Risk metrics and alerts
- System control (start/stop)

The API server is built with Flask and runs on port 8081 by default. It enables integration with external applications and services.

#### API Documentation

When running, the API documentation is available at http://localhost:8081/



## Operation Modes

### Live Trading

The system can operate in live trading mode, continuously updating market data, generating signals, and updating the portfolio.

### Backtesting

The system supports backtesting mode for evaluating strategies using historical data.

### Model Training

The system includes a model training mode for updating AI models with new data.

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up API keys in environment variables or config file

### Configuration

The system is configured via the `config.yaml` file, which includes settings for:

- System parameters
- Trading parameters
- Database configuration
- API keys
- AI model parameters
- Strategy parameters

### Running the System

```bash
# Run in live trading mode
python main.py --mode run

# Run in backtesting mode
python main.py --mode backtest --start-date 2022-01-01 --end-date 2022-12-31 --initial-capital 100000

# Run in model training mode
python main.py --mode train
```

## Project Structure

```
AI-HedgeFund/
├── main.py                 # Main application entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Dependencies
├── src/
│   ├── agents/             # Agent implementations
│   │   ├── agent_base.py
│   │   ├── warren_buffett_agent.py
│   │   ├── peter_lynch_agent.py
│   │   └── ...
│   ├── core/               # Core components
│   │   ├── data_manager.py
│   │   ├── strategy_manager.py
│   │   ├── portfolio_manager.py
│   │   └── ...
│   ├── strategies/         # Strategy implementations
│   │   ├── warren_buffett_ai.py
│   │   ├── economic_macro_strategy.py
│   │   └── ...
│   └── utils/              # Utility modules
│       ├── portfolio_optimizer.py
│       ├── risk_metrics.py
│       └── ...
├── data/                   # Data storage
├── logs/                   # Log files
└── models/                 # Trained models
```
