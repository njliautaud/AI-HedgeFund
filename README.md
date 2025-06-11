# AI Hedge Fund

A streamlined, rule-based AI investment system featuring specialized analysis agents and an LLM-based decision maker. This system uses a combination of fundamental, technical, sentiment, and market analysis to make informed investment decisions.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- Gemini API key (for LLM functionality)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/njliautaud/AI-HedgeFund.git
   cd AI-HedgeFund
   ```

2. Create and activate a virtual environment:
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

4. Set up your Gemini API key:
   ```bash
   # On Windows
   setx GEMINI_API_KEY "your-api-key-here"
   
   # On macOS/Linux
   export GEMINI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```

### Running the Example

To analyze a set of stocks and get investment recommendations:

```bash
python example.py
```

This will analyze several major stocks and provide detailed investment recommendations based on the combined analysis of multiple specialized agents.

### Architecture

The system consists of several specialized agents that analyze different aspects of a stock:

1. **Fundamental Agent**: Analyzes company financials, valuation metrics, and growth potential.
2. **Technical Agent**: Examines price patterns, trends, and technical indicators.
3. **Sentiment Agent**: Assesses market sentiment from news and social media.
4. **Market Agent**: Analyzes overall market conditions and sector performance.
5. **LLM Manager**: Synthesizes information from all agents to make final investment decisions.

### Configuration

You can customize the behavior of the agents by modifying the `config.yaml` file. Key settings include:

- Investment rules and scoring weights
- Technical indicator parameters
- Market analysis preferences
- API rate limiting and caching settings

## Example Output

```
Analyzing AAPL...

================================================================================
Analysis for AAPL - Apple Inc.
Sector: Technology
Current Price: $175.34

Fundamental Analysis:
- Score: 82.5/100
- Summary: Apple Inc. shows strong fundamentals with a score of 82.5/100...

Technical Analysis:
- Score: 68.3/100
- Summary: Mixed technical signals with a neutral to slightly bullish bias...

Sentiment Analysis:
- Score: 72.1/100
- Summary: Sentiment analysis shows positive sentiment based on 15 recent articles...

Market Context:
- Market conditions are favorable with major indices trending upward.
- Technology sector is outperforming the broader market.

Investment Decision:
[LLM-generated investment recommendation with reasoning]
================================================================================
```

## Requirements

- Python 3.9+
- yfinance
- pandas
- numpy
- google-generativeai
- python-dotenv

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and informational purposes only. It is not intended as financial advice, and you should not rely on it as such. Always do your own research and consider your financial situation before making investment decisions.
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
## Key Features

- **Rule-based Analysis**: Transparent, explainable decision-making process
- **Modular Design**: Easy to add or modify analysis components
- **Multi-timeframe Analysis**: Combines short, medium, and long-term perspectives
- **Risk Management**: Built-in position sizing and stop-loss mechanisms
- **Customizable**: Adjust parameters and rules to match your investment style

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
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
