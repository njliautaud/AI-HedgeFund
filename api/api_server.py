#!/usr/bin/env python3
"""
AI Hedge Fund - API Server

Provides the backend API endpoints for the AI Hedge Fund system.
Interacts with various managers (AgentManager, AIModelManager, etc.)
to fulfill requests from the Dash dashboard or other clients.
"""

import os
import sys
from pathlib import Path
import time

from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS

# Add project root to path to allow importing project modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_manager import ConfigManager
# Placeholder for actual manager imports - these would be initialized and used by the API
# from core.agent_manager import AgentManager
# from core.ai_model_manager import AIModelManager
# from core.portfolio_manager import PortfolioManager
# from core.data_manager import DataManager
# from core.system_controller import SystemController

# --- Global Variables & Initialization ---
config_path = Path(__file__).parent.parent / 'config.yaml'
if not config_path.exists():
    # Create a dummy config.yaml if it doesn't exist for standalone testing
    dummy_config_content = """
ui:
  dashboard_port: 8080
  api_port: 8081
api:
  host: "0.0.0.0"
  port: 8081
system:
  mode: "paper" # paper, live, backtest, training
"""
    with open(config_path, 'w') as f:
        f.write(dummy_config_content)
    print(f"Created dummy {config_path} for API server testing.")

app_config = ConfigManager.load_config(config_path)

API_PORT = app_config.get('api', {}).get('port', 8081)
API_HOST = app_config.get('api', {}).get('host', '0.0.0.0')

# Initialize Flask App
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing dashboard to connect

# API Blueprint for versioning
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')

# --- Placeholder Data & Mock Managers (Replace with actual manager interactions) ---
MOCK_AGENTS = [
    {
        'id': 'agent_001', 
        'name': 'Fundamental Analysis Agent', 
        'status': 'active', 
        'agent_class': 'FundamentalAnalysisAgent', 
        'config': {
            'max_pe_ratio': 30.0,
            'max_peg_ratio': 2.0,
            'min_current_ratio': 1.5,
            'max_debt_to_equity': 1.0,
            'margin_of_safety': 0.2
        }, 
        'metrics': {'trades_made': 8, 'pnl': 1200.50}
    },
    {
        'id': 'agent_002', 
        'name': 'Warren Buffett Value Agent', 
        'status': 'active', 
        'agent_class': 'LegendaryInvestorAgent', 
        'config': {'risk_aversion': 0.5}, 
        'metrics': {'trades_made': 10, 'pnl': 1500.75}
    },
    {
        'id': 'agent_003', 
        'name': 'Ray Dalio All Weather Agent', 
        'status': 'idle', 
        'agent_class': 'PortfolioStrategyAgent', 
        'config': {'rebalance_threshold': 0.02}, 
        'metrics': {'trades_made': 5, 'pnl': 800.20}
    },
    {
        'id': 'agent_004', 
        'name': 'Momentum Trader Agent', 
        'status': 'active', 
        'agent_class': 'TechnicalAgent', 
        'config': {'momentum_window': 20}, 
        'metrics': {'trades_made': 25, 'pnl': -300.50}
    }
]

MOCK_MODELS = [
    {'id': 'model_v1_warren', 'name': 'Warren Buffett Model v1.0', 'type': 'LegendaryInvestorAgent', 'status': 'trained', 'accuracy': 0.78},
    {'id': 'model_v1_momentum', 'name': 'Momentum Model v1.2', 'type': 'TechnicalAgent', 'status': 'training', 'progress': 0.65}
]

# --- Agent Endpoints ---
@api_v1.route('/agents', methods=['GET'])
def get_agents():
    """Lists all available agents."""
    # In a real app, this would fetch from AgentManager
    return jsonify(MOCK_AGENTS)

@api_v1.route('/agents/<string:agent_id>', methods=['GET'])
def get_agent_details(agent_id):
    """Gets detailed information for a specific agent."""
    agent = next((agent for agent in MOCK_AGENTS if agent['id'] == agent_id), None)
    if agent:
        return jsonify(agent)
    return jsonify({'error': 'Agent not found'}), 404

@api_v1.route('/agents/<string:agent_id>/analyze_ticker', methods=['POST'])
def analyze_ticker_with_agent(agent_id):
    """Analyzes a ticker using a specific agent."""
    data = request.get_json()
    ticker = data.get('ticker')
    
    agent = next((agent for agent in MOCK_AGENTS if agent['id'] == agent_id), None)
    if not agent:
        return jsonify({'error': 'Agent not found'}), 404
    
    # Handle fundamental analysis agent
    if agent_id == 'agent_001':
        # In a real implementation, this would call the actual agent's analyze method
        # For now, we'll return a mock response
        import random
        
        pe_ratio = round(random.uniform(5, 40), 2)
        peg_ratio = round(random.uniform(0.5, 3.0), 2)
        roe = round(random.uniform(0.05, 0.30), 2)
        
        result = {
            'ticker': ticker.upper(),
            'agent_id': agent_id,
            'agent_name': agent.get('name'),
            'recommendation': 'BUY' if pe_ratio < 20 and peg_ratio < 2.0 and roe > 0.15 else 'HOLD',
            'analysis': {
                'company_name': f"{ticker.upper()} Inc.",
                'current_price': round(random.uniform(50, 500), 2),
                'target_price': round(random.uniform(40, 600), 2),
                'fundamentals': {
                    'pe_ratio': pe_ratio,
                    'peg_ratio': peg_ratio,
                    'roe': roe,
                    'current_ratio': round(random.uniform(1.0, 3.0), 2),
                    'debt_to_equity': round(random.uniform(0.1, 2.0), 2),
                    'dividend_yield': round(random.uniform(0.0, 0.05), 4)
                },
                'reasoning': f"Analysis based on fundamental metrics: P/E {pe_ratio}, PEG {peg_ratio}, ROE {roe*100}%"
            }
        }
    else:
        # For other agents, return a simpler response
        result = {
            'ticker': ticker.upper(),
            'agent_id': agent_id,
            'agent_name': agent.get('name'),
            'recommendation': 'BUY' if hash(ticker + agent_id) % 2 == 0 else 'HOLD',
            'analysis': {
                'reasoning': f'Mock analysis based on proprietary signals for {ticker.upper()} by {agent.get("name")}.'
            }
        }
    
    return jsonify(result)

@api_v1.route('/fundamental-analysis/<string:ticker>', methods=['GET'])
def get_fundamental_analysis(ticker):
    """Get fundamental analysis for a ticker using the Fundamental Analysis Agent."""
    # In a real implementation, this would call the actual agent's analyze method
    # For now, we'll return a mock response
    import random
    
    pe_ratio = round(random.uniform(5, 40), 2)
    peg_ratio = round(random.uniform(0.5, 3.0), 2)
    roe = round(random.uniform(0.05, 0.30), 2)
    
    result = {
        'ticker': ticker.upper(),
        'company_name': f"{ticker.upper()} Inc.",
        'current_price': round(random.uniform(50, 500), 2),
        'target_price': round(random.uniform(40, 600), 2),
        'recommendation': 'BUY' if pe_ratio < 20 and peg_ratio < 2.0 and roe > 0.15 else 'HOLD',
        'fundamentals': {
            'pe_ratio': pe_ratio,
            'peg_ratio': peg_ratio,
            'roe': roe,
            'current_ratio': round(random.uniform(1.0, 3.0), 2),
            'debt_to_equity': round(random.uniform(0.1, 2.0), 2),
            'dividend_yield': round(random.uniform(0.0, 0.05), 4),
            'sector': random.choice(['Technology', 'Healthcare', 'Financial', 'Consumer Cyclical', 'Industrial']),
            'market_cap': random.choice(['Small Cap', 'Mid Cap', 'Large Cap']),
            'revenue_growth': round(random.uniform(-0.1, 0.3), 2),
            'earnings_growth': round(random.uniform(-0.2, 0.4), 2)
        },
        'valuation_metrics': {
            'pe_valuation': round(pe_ratio * 1.1, 2),
            'dcf_valuation': round(random.uniform(50, 600), 2),
            'comparable_companies_valuation': round(random.uniform(40, 550), 2)
        },
        'analysis': {
            'strengths': [
                f"Strong return on equity ({roe*100:.1f}%)",
                "Consistent earnings growth",
                "Strong competitive position"
            ],
            'risks': [
                "Market volatility",
                "Regulatory changes",
                "Competitive pressures"
            ],
            'summary': f"{ticker.upper()} shows solid fundamentals with a reasonable valuation. The company demonstrates strong profitability and growth potential, though investors should be mindful of industry risks."
        }
    }
    
    return jsonify(result)

# --- Model Training Endpoints ---
@api_v1.route('/models/train', methods=['POST'])
def train_model():
    """Initiates training for a new AI model."""
    data = request.get_json()
    model_name = data.get('model_name') # This is 'model_selection' value from dashboard
    epochs = data.get('epochs')
    batch_size = data.get('batch_size')
    analysis_type = data.get('analysis_type')

    if not all([model_name, epochs, batch_size, analysis_type]):
        return jsonify({'error': 'Missing training parameters'}), 400

    # Mock training process
    time.sleep(2) # Simulate training time
    mock_history = {'loss': [0.5, 0.3, 0.2], 'val_loss': [0.6, 0.4, 0.3], 'accuracy': [0.7, 0.8, 0.85]}
    mock_metrics = {'final_accuracy': 0.85, 'final_loss': 0.2, 'training_time_sec': 2}
    
    # Add new model to mock list (or update if exists)
    new_model_id = f"model_{model_name.split('_')[-1]}_{int(time.time()) % 1000}"
    MOCK_MODELS.append({
        'id': new_model_id,
        'name': f"{model_name.replace('_', ' ').title()} (Trained)", 
        'type': model_name, 
        'status': 'trained', 
        'accuracy': mock_metrics['final_accuracy']
    })

    return jsonify({
        'message': f'Model training for {model_name} initiated successfully. New model ID: {new_model_id}',
        'model_id': new_model_id,
        'history': mock_history,
        'metrics': mock_metrics
    })

@api_v1.route('/models', methods=['GET'])
def get_models():
    """Lists all trained/training models."""
    return jsonify(MOCK_MODELS)

# --- System Control Endpoints (Placeholders) ---
@api_v1.route('/system/status', methods=['GET'])
def get_system_status():
    # In a real app, this would fetch from SystemController
    return jsonify({'status': 'running', 'mode': app_config.get('system', {}).get('mode', 'paper'), 'uptime_seconds': int(time.time() % 10000)})

@api_v1.route('/system/start', methods=['POST'])
def start_system():
    # SystemController.start()
    return jsonify({'status': 'System start requested', 'message': 'System is now starting.'})

@api_v1.route('/system/stop', methods=['POST'])
def stop_system():
    # SystemController.stop()
    return jsonify({'status': 'System stop requested', 'message': 'System is now stopping.'})

@api_v1.route('/system/mode', methods=['POST'])
def set_system_mode():
    data = request.get_json()
    mode = data.get('mode')
    if mode not in ['live', 'paper', 'backtest', 'training']:
        return jsonify({'error': 'Invalid mode specified'}), 400
    # SystemController.set_mode(mode)
    # Update config (if persistent change is desired, or just in-memory for SystemController)
    app_config['system']['mode'] = mode # Example of in-memory update
    return jsonify({'status': f'System mode set to {mode}', 'new_mode': mode})

# Emerging Markets Endpoints
@api_v1.route('/emerging-markets', methods=['GET'])
def get_emerging_markets():
    """Get emerging markets data."""
    try:
        from src.data.emerging_markets import get_emerging_markets_summary
        summary = get_emerging_markets_summary()
        return jsonify({
            'status': 'success',
            'data': summary
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Register blueprint
app.register_blueprint(api_v1)

# --- Main Execution ---
if __name__ == '__main__':
    print(f"AI Hedge Fund API Server starting on http://{API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=True)