#!/usr/bin/env python3
"""
AI Hedge Fund - Dash Web Dashboard

A sleek, minimalist web dashboard for the AI Hedge Fund system using Dash.
Provides a modern, responsive interface for monitoring and controlling the system.
"""

import os
import sys
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
import requests # Added for API calls

# Dash imports
import dash
from dash import dcc, html, Input, Output, State, callback, callback_context # Added State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class DashDashboard:
    """
    Web dashboard for the AI Hedge Fund system using Dash.
    Provides a user-friendly interface to monitor and control the system.
    """
    
    def __init__(self, config, portfolio_manager=None, strategy_manager=None, risk_manager=None, data_manager=None, agent_manager=None):
        """
        Initialize the dashboard with system components.
        
        Args:
            config: System configuration
            portfolio_manager: Portfolio management component
            strategy_manager: Strategy management component
            risk_manager: Risk management component (can be None)
            data_manager: Data management component
            agent_manager: Agent management component
        """
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.strategy_manager = strategy_manager
        self.risk_manager = risk_manager # May be None
        self.data_manager = data_manager
        self.agent_manager = agent_manager 
        self.ai_model_manager = agent_manager.ai_model_manager if agent_manager else None
        
        self.dashboard_port = self.config.get('ui', {}).get('dashboard_port', 8080)
        self.api_base_url = f"http://localhost:{self.config.get('ui', {}).get('api_port', 8081)}/api/v1" # Added API base URL
        self.dashboard_thread = None
        self.running = False
        
        # Initialize the Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME], # Changed theme for a more modern look
            title="AI Hedge Fund Dashboard",
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
        )
        self.server = self.app.server # Expose server for Gunicorn/Waitress
        
        # Set up the layout using URL routing
        self.app.layout = self._create_main_layout() # Changed to main layout for routing
        
        # Set up callbacks
        self._setup_navigation_callbacks() # For page routing
        self._setup_callbacks() # General system control callbacks
        self._setup_agent_callbacks() # Agent-specific callbacks
        self._setup_model_training_callbacks() # Model training callbacks
            
        # Add other specific callback setups here (e.g., portfolio, strategy)
    
    def _run_dashboard(self):
        """
        Run the Dash dashboard.
        This method is run in a separate thread.
        """
        try:
            logger.info(f"Starting dashboard server on port {self.dashboard_port}")
            self.app.run_server(debug=False, port=self.dashboard_port, host='0.0.0.0')
        except Exception as e:
            logger.error(f"Error in dashboard server: {e}")
            self.running = False
    
    async def start(self):
        """
        Start the dashboard in a separate thread.
        """
        if self.running:
            logger.warning("Dashboard is already running")
            return
            
        try:
            self.running = True
            # Create a thread to run the dashboard
            self.dashboard_thread = threading.Thread(
                target=self._run_dashboard,
                daemon=True,
                name="DashboardThread"
            )
            self.dashboard_thread.start()
            logger.info("Dashboard started")
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            self.running = False
            raise
        
    async def stop(self):
        """
        Stop the dashboard.
        """
        self.running = False
        # The thread will terminate when the main process exits
    
    def _create_main_layout(self):
        """
        Create the main layout of the dashboard with navigation and content area.
        
        Returns:
            dash.html.Div: The main layout component
        """
        # Define the navigation bar
        navbar = dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Overview", href="/", active="exact")),
                dbc.NavItem(dbc.NavLink("Portfolio", href="/portfolio", active="exact")),
                dbc.NavItem(dbc.NavLink("Strategies", href="/strategies", active="exact")),
                dbc.NavItem(dbc.NavLink("Risk", href="/risk", active="exact")),
                dbc.NavItem(dbc.NavLink("Models", href="/models", active="exact")),
                dbc.NavItem(dbc.NavLink("Agents", href="/agents", active="exact")),
                dbc.NavItem(dbc.NavLink("Settings", href="/settings", active="exact")),
            ],
            brand="AI Hedge Fund",
            brand_href="/",
            color="primary",
            dark=True,
            className="mb-4"
        )
        
        # Define the content area that will be updated based on URL
        content = html.Div(id="page-content", className="container")
        
        # Combine navbar and content with URL routing
        layout = html.Div([
            dcc.Location(id="url"),
            navbar,
            content,
            # Add hidden div for storing intermediate values
            html.Div(id="intermediate-value", style={"display": "none"}),
            # Add interval for auto-updating data
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # in milliseconds, update every minute
                n_intervals=0
            )
        ])
        
        return layout
        
    def _run_dashboard(self):
        """
        Run the Dash dashboard.
        """
        self.app.run(debug=False, port=self.dashboard_port, host='0.0.0.0')
    
    def _create_model_training_chart(self, model_id, metrics):
        """
        Create a chart showing model training metrics over epochs.
        
        Args:
            model_id: Identifier for the model
            metrics: Dictionary containing training metrics
            
        Returns:
            Plotly figure object
        """
        # Convert metrics to DataFrame for plotting
        epochs = list(range(1, len(metrics['loss']) + 1))
        training_data = pd.DataFrame({
            'Epoch': epochs,
            'Training Loss': metrics['loss'],
            'Validation Loss': metrics['val_loss'],
            'Training Accuracy': metrics['accuracy'],
            'Validation Accuracy': metrics['val_accuracy']
        })
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add traces for loss (left y-axis)
        fig.add_trace(go.Scatter(
            x=training_data['Epoch'],
            y=training_data['Training Loss'],
            name='Training Loss',
            line=dict(color='#1f77b4')
        ))
        fig.add_trace(go.Scatter(
            x=training_data['Epoch'],
            y=training_data['Validation Loss'],
            name='Validation Loss',
            line=dict(color='#1f77b4', dash='dash')
        ))
        
        # Add traces for accuracy (right y-axis)
        fig.add_trace(go.Scatter(
            x=training_data['Epoch'],
            y=training_data['Training Accuracy'],
            name='Training Accuracy',
            line=dict(color='#ff7f0e'),
            yaxis='y2'
        ))
        fig.add_trace(go.Scatter(
            x=training_data['Epoch'],
            y=training_data['Validation Accuracy'],
            name='Validation Accuracy',
            line=dict(color='#ff7f0e', dash='dash'),
            yaxis='y2'
        ))
        
        # Set up layout with two y-axes
        fig.update_layout(
            title=f'Training Metrics for {model_id}',
            xaxis=dict(title='Epoch'),
            yaxis=dict(
                title='Loss',
                titlefont=dict(color='#1f77b4'),
                tickfont=dict(color='#1f77b4')
            ),
            yaxis2=dict(
                title='Accuracy',
                titlefont=dict(color='#ff7f0e'),
                tickfont=dict(color='#ff7f0e'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )
        
        return fig
    
    def _setup_layout(self):
        """
        Set up the Dash app layout.
        """
        # Create a modern, sleek layout with sidebar and main content area
        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        # Sidebar
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.I(className="fas fa-brain fa-2x me-2"), # Changed icon
                                        html.H2("AI Hedge Fund", className="display-6"),
                                    ],
                                    className="d-flex align-items-center mt-3 mb-4"
                                ),
                                html.Hr(),
                                dbc.Nav(
                                    [
                                        dbc.NavLink([html.I(className="fas fa-tachometer-alt me-2"), "Overview"], href="/", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-users-cog me-2"), "Agent Control"], href="/agents", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-cogs me-2"), "Model Training"], href="/models", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-chart-pie me-2"), "Portfolio"], href="/portfolio", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-lightbulb me-2"), "Strategies"], href="/strategies", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-database me-2"), "Market Data"], href="/market-data", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-shield-alt me-2"), "Risk Metrics"], href="/risk", active="exact"),
                                    ],
                                    vertical=True,
                                    pills=True,
                                    className="mb-4"
                                ),
                                html.Hr(),
                                html.Div(
                                    [
                                        html.H5("System Information", className="mt-4 mb-2"),
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.P(f"Dashboard: http://localhost:{self.dashboard_port}"),
                                                    html.P(f"API Server: {self.api_base_url.replace('/api/v1', '')}")
                                                ]
                                            ),
                                            className="mb-3"
                                        )
                                    ]
                                )
                            ],
                            width=3,
                            className="bg-light p-4 h-100 shadow-sm"
                        ),
                        
                        # Main content area - content will be rendered here by callbacks
                        dbc.Col(
                            [
                                dcc.Location(id='url', refresh=False),
                                html.Div(id='page-content', className="p-4") 
                            ],
                            width=9,
                            className="h-100 overflow-auto" # Ensure content area scrolls if needed
                        )
                    ],
                    className="vh-100 m-0" # Ensure full height and no margin
                )
            ],
            fluid=True,
            className="p-0"
        )

    def _render_page_content(self, pathname):
        """Renders the content for the selected page based on the URL pathname."""
        if not pathname or pathname == "/":
            return self._create_overview_page()
        elif pathname == "/agents":
            return self._create_agent_control_page()
        elif pathname == "/models":
            return self._create_model_training_page()
        elif pathname == "/portfolio":
            return self._create_portfolio_page() # Placeholder
        elif pathname == "/strategies":
            return self._create_strategies_page() # Placeholder
        elif pathname == "/market-data":
            return self._create_market_data_page() # Placeholder
        elif pathname == "/risk":
            return self._create_risk_metrics_page() # Placeholder
        # Add more pages here
        return html.Div(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname if pathname else 'None'} was not recognised...")
            ],
            className="p-3"
        )
    
    def _create_overview_page(self):
        """Creates the layout for the Overview page."""
        # This combines elements from the old sidebar and main content for system control
        return html.Div([
            html.H3("System Overview & Control", className="mb-4"),
            dbc.Row([
                dbc.Col([
                    html.H5("System Status", className="mb-3"),
                    html.Div([
                        html.P("Status: ", className="d-inline fw-bold"),
                        html.P("Running", id="system-status", className="d-inline text-success") # Example status
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(dbc.Button("Start System", id="start-button", color="success", className="w-100 mb-2")), # API call needed
                        dbc.Col(dbc.Button("Stop System", id="stop-button", color="danger", className="w-100 mb-2"))    # API call needed
                    ], className="mb-4"),
                    html.H5("System Mode", className="mb-2"),
                    dbc.RadioItems(
                        options=[
                            {"label": "Live Trading", "value": "live"},
                            {"label": "Paper Trading", "value": "paper"},
                            {"label": "Backtesting", "value": "backtest"},
                            {"label": "Training", "value": "training"}
                        ],
                        value="paper", # Default mode
                        id="mode-selection",
                        className="mb-3"
                    ), # API call needed to set mode
                    html.Div(
                        [
                            html.H6("Backtesting Period", className="mt-3 mb-2"),
                            dbc.Label("Start Date"),
                            dcc.DatePickerSingle(id="backtest-start-date", date=datetime.now() - timedelta(days=365), className="mb-2 w-100"),
                            dbc.Label("End Date"),
                            dcc.DatePickerSingle(id="backtest-end-date", date=datetime.now(), className="mb-2 w-100"),
                            dbc.Label("Initial Capital"),
                            dbc.Input(id="initial-capital", type="number", value=100000, min=10000, step=10000, className="w-100")
                        ],
                        id="backtest-settings",
                        style={"display": "none"} # Initially hidden
                    )
                ], md=6),
                dbc.Col([
                    html.H5("Recent Activity/Logs", className="mb-3"),
                    dcc.Textarea(
                        id='system-logs-overview',
                        value='System initialized.\nMonitoring market data...\nAgent X performed action Y.',
                        style={'width': '100%', 'height': 300},
                        readOnly=True,
                        className="mb-3"
                    ), # To be populated by API call or WebSocket
                    # Placeholder for key metrics or alerts
                    html.H5("Key Metrics", className="mt-4 mb-3"),
                    dbc.CardGroup([
                        dbc.Card(dbc.CardBody([html.H6("Active Agents"), html.H4(id="active-agents-metric", children="N/A")])), # API call
                        dbc.Card(dbc.CardBody([html.H6("Trades Today"), html.H4(id="trades-today-metric", children="N/A")])), # API call
                        dbc.Card(dbc.CardBody([html.H6("Current P&L"), html.H4(id="current-pnl-metric", children="N/A")])) # API call
                    ])
                ], md=6)
            ])
        ])

    def _create_agent_control_page(self):
        """Creates the layout for the Agent Control page."""
        return html.Div([
            html.H3("Agent Control & Ticker Analysis", className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Select Agent"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='agent-selector-dropdown',
                                options=[], # To be populated by callback
                                placeholder="Select an Agent",
                                className="mb-3"
                            ),
                            dcc.Loading(id="loading-agent-details", children=[html.Div(id='agent-details-output')])
                        ])
                    ], className="mb-3")
                ], md=5),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Ticker Analysis"),
                        dbc.CardBody([
                            dcc.Input(id="ticker-input", type="text", placeholder="Enter Ticker (e.g., AAPL)", className="mb-2 w-100"),
                            dcc.Dropdown(
                                id='agent-for-ticker-analysis-dropdown',
                                options=[], # To be populated by callback
                                placeholder="Select Agent for Analysis",
                                className="mb-3"
                            ),
                            dbc.Button("Analyze Ticker", id="analyze-ticker-button", color="primary", className="w-100"),
                            dcc.Loading(id="loading-ticker-analysis", children=[html.Div(id="ticker-analysis-results", className="mt-3")])
                        ])
                    ])
                ], md=7)
            ])
        ])

    def _create_model_training_page(self):
        """Creates the layout for the Model Training page."""
        return html.Div([
            html.H3("AI Model Training & Analysis", className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Train New Model"),
                        dbc.CardBody([
                            dbc.Label("Select Model Type/Agent Base"),
                            dcc.Dropdown(
                                id="model-selection", 
                                options=[
                                    # These should ideally be fetched from an API endpoint listing available model blueprints/agent types
                                    {"label": "Warren Buffett Value Investor", "value": "legendary_investors_warren_buffett"},
                                    {"label": "Peter Lynch Growth Investor", "value": "legendary_investors_peter_lynch"},
                                    {"label": "Benjamin Graham Deep Value", "value": "legendary_investors_benjamin_graham"},
                                    {"label": "Ray Dalio All Weather", "value": "legendary_investors_ray_dalio"},
                                    {"label": "Momentum Strategy", "value": "technical_momentum"},
                                    {"label": "Mean Reversion Strategy", "value": "technical_mean_reversion"}
                                ], 
                                value="legendary_investors_warren_buffett", 
                                className="mb-3"
                            ),
                            dbc.Row([
                                dbc.Col([dbc.Label("Epochs"), dbc.Input(id="training-epochs", type="number", value=50, min=10, max=1000, step=10)], width=6, className="mb-2"),
                                dbc.Col([dbc.Label("Batch Size"), dbc.Input(id="training-batch-size", type="number", value=32, min=8, max=256, step=8)], width=6, className="mb-2")
                            ]),
                            dbc.Label("Analysis Type for Training Data"),
                            dbc.RadioItems(
                                id="analysis-type",
                                options=[
                                    {"label": "Fundamental", "value": "fundamental"},
                                    {"label": "Technical", "value": "technical"},
                                    {"label": "Sentiment", "value": "sentiment"},
                                    {"label": "All", "value": "all"}
                                ],
                                value="all",
                                inline=True,
                                className="mb-3"
                            ),
                            dbc.Button("Train Model", id="train-model-button", color="primary", className="w-100")
                        ])
                    ])
                ], md=5),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Results & Metrics"),
                        dbc.CardBody([
                            dcc.Loading(id="loading-model-training", children=[
                                html.Div(id="model-training-status"), 
                                html.Div(id="model-training-history-graph"), 
                                html.Div(id="model-metrics-display", className="mt-3")
                            ])
                        ])
                    ])
                ], md=7)
            ]),
            html.Hr(className="my-4"),
            html.H4("Existing Models/Agents", className="mb-3"),
            # Placeholder for a table or list of existing trained models/agents, with options to view details, deploy, or delete
            html.Div(id="existing-models-list", children="[Agent/Model listing will appear here - API call needed]") 
        ])

    # Placeholder methods for other pages - to be implemented
    def _create_portfolio_page(self): return html.Div([html.H3("Portfolio Management"), html.P("Portfolio details and charts will go here.")])
    def _create_strategies_page(self): return html.Div([html.H3("Strategy Management"), html.P("Strategy allocation and performance will go here.")])
    def _create_market_data_page(self): return html.Div([html.H3("Market Data"), html.P("Market indices and economic indicators will go here.")])
    def _create_risk_metrics_page(self): return html.Div([html.H3("Risk Metrics"), html.P("Risk metrics and heatmaps will go here.")])

    def _setup_navigation_callbacks(self):
        """Sets up callbacks for page navigation based on URL."""
        @self.app.callback(
            Output("page-content", "children"),
            [Input("url", "pathname")],
            prevent_initial_call=True
        )
        def display_page(pathname):
            """Display the appropriate page based on the URL pathname."""
            if pathname is None:
                return self._render_page_content("/")  # Default to home
            return self._render_page_content(pathname)

    def _setup_callbacks(self):
        """Set up general Dash callbacks for system control and UI updates."""
        # Callback to show/hide backtesting settings
        @self.app.callback(
            Output("backtest-settings", "style"),
            Input("mode-selection", "value"),
            prevent_initial_call=True
        )
        def toggle_backtest_settings(mode):
            if mode == "backtest":
                return {"display": "block"}
            return {"display": "none"}

        # System control callbacks
        @self.app.callback(
            Output("system-status", "children"),
            Input("start-button", "n_clicks"),
            prevent_initial_call=True
        )
        def handle_start_system(n_clicks):
            if not n_clicks:
                return dash.no_update
                
            try:
                # Try to use API if available
                if hasattr(self, 'api_base_url'):
                    response = requests.post(f"{self.api_base_url}/system/start")
                    if response.status_code == 200:
                        return "Running"
                    return "Error Starting"
                # Fallback to direct method call if API not available
                elif hasattr(self, 'portfolio_manager'):
                    # This is a placeholder - implement actual system start logic
                    logger.info("Starting system directly (no API)")
                    return "Running"
                return "Error: No API or direct method available"
            except Exception as e:
                logger.error(f"Error starting system: {e}")
                return f"Error: {str(e)}"
        
        # Add other general callbacks (stop, mode change, etc.) here, interacting with self.api_base_url
        pass

    def _setup_agent_callbacks(self):
        """Sets up callbacks related to agent control and ticker analysis."""
        
        # Callback to populate agent selector dropdowns
        @self.app.callback(
            [Output('agent-selector-dropdown', 'options'),
             Output('agent-for-ticker-analysis-dropdown', 'options'),
             Output('agent-selector-dropdown', 'value'), # To pre-select if only one agent or first agent
             Output('agent-for-ticker-analysis-dropdown', 'value')],
            [Input('url', 'pathname')]  # Trigger when page loads or URL changes to agent control page
        )
        def populate_agent_selectors(pathname):
            if pathname == "/agents": # Only fetch if on the relevant page
                try:
                    response = requests.get(f"{self.api_base_url}/agents") # API call to list agents
                    response.raise_for_status()
                    agents = response.json()
                    if not agents:
                        return [], [], None, None
                    options = [{'label': agent.get('name', agent.get('id', 'Unnamed Agent')), 'value': agent['id']} for agent in agents]
                    # Pre-select the first agent if available
                    first_agent_id = options[0]['value'] if options else None
                    return options, options, first_agent_id, first_agent_id
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching agents: {e}")
                    # Optionally, return an error message to be displayed in the UI
                    return [], [], None, None
                except Exception as e:
                    print(f"An unexpected error occurred while fetching agents: {e}")
                    return [], [], None, None
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update # Don't update if not on agent control page

        # Callback to display agent details
        @self.app.callback(
            Output('agent-details-output', 'children'),
            [Input('agent-selector-dropdown', 'value')],
            [State('url', 'pathname')] 
        )
        def display_agent_details(agent_id, pathname):
            if not agent_id or pathname != "/agents":
                if pathname == "/agents" and not agent_id:
                     return dbc.Alert("Select an agent to see details.", color="info", className="mt-2")
                return dash.no_update
            try:
                response = requests.get(f"{self.api_base_url}/agents/{agent_id}") # API call for agent details
                response.raise_for_status()
                agent_details = response.json()
                
                # Nicely formatted agent details
                card_content = [
                    html.H5(agent_details.get('name', 'N/A'), className="card-title"),
                    html.P(f"ID: {agent_details.get('id')}"),
                    html.P(f"Status: {agent_details.get('status', 'N/A')}"),
                    html.P(f"Type/Class: {agent_details.get('agent_class', 'N/A')}"),
                    html.H6("Configuration:", className="mt-3"),
                    dcc.Markdown(f"```json\n{json.dumps(agent_details.get('config', {}), indent=2)}\n```"),
                    # Add more details as available from your API
                    html.H6("Live Metrics (Example):", className="mt-3"),
                    html.P(f"Trades Made: {agent_details.get('metrics', {}).get('trades_made', 'N/A')}"),
                    html.P(f"Current P&L: {agent_details.get('metrics', {}).get('pnl', 'N/A')}"),
                    # Add buttons for agent actions if applicable (e.g., start/stop agent, update config)
                    # dbc.Button("Stop Agent", color="warning", className="mt-2 me-2"),
                    # dbc.Button("Update Config", color="info", className="mt-2"),
                ]
                return dbc.Card(dbc.CardBody(card_content), className="mt-2 shadow-sm")
            except requests.exceptions.RequestException as e:
                return dbc.Alert(f"Error fetching agent details: {e.response.text if e.response else str(e)}", color="danger", className="mt-2")
            except Exception as e:
                return dbc.Alert(f"An unexpected error occurred: {e}", color="danger", className="mt-2")

        # Callback for ticker analysis
        @self.app.callback(
            Output('ticker-analysis-results', 'children'),
            [Input('analyze-ticker-button', 'n_clicks')],
            [State('ticker-input', 'value'),
             State('agent-for-ticker-analysis-dropdown', 'value'),
             State('url', 'pathname')]
        )
        def analyze_ticker(n_clicks, ticker, agent_id, pathname):
            if not n_clicks or pathname != "/agents":
                return dash.no_update
            
            if not ticker or not agent_id:
                return dbc.Alert("Please enter a ticker and select an agent for analysis.", color="warning", className="mt-2")
            
            try:
                # API endpoint for ticker analysis by a specific agent
                # This might be /agents/{agent_id}/analyze or /analysis/ticker with agent_id in payload
                # Adjust based on your API design in api_server.py
                payload = {"ticker": ticker} # Agent ID is in the URL or could be part of payload
                response = requests.post(f"{self.api_base_url}/agents/{agent_id}/analyze_ticker", json=payload)
                response.raise_for_status()
                analysis_result = response.json()
                
                # Format the result for display
                result_display = [
                    html.H5(f"Analysis for {ticker.upper()} by Agent {agent_details.get('name', agent_id) if 'agent_details' in locals() else agent_id}", className="mt-3"), # Requires agent_details to be available or fetched
                    html.P(f"Recommendation: {analysis_result.get('recommendation', 'N/A')}"),
                    html.P(f"Confidence: {analysis_result.get('confidence', 'N/A')}"),
                    html.H6("Details:", className="mt-2"),
                    dcc.Markdown(f"```json\n{json.dumps(analysis_result.get('details', {}), indent=2)}\n```")
                ]
                return dbc.Card(dbc.CardBody(result_display), className="mt-2 shadow-sm")
            except requests.exceptions.RequestException as e:
                error_message = f"API Error during ticker analysis: {e.response.text if e.response else str(e)}"
                return dbc.Alert(error_message, color="danger", className="mt-2")
            except Exception as e:
                return dbc.Alert(f"An unexpected error occurred: {e}", color="danger", className="mt-2")

    def _setup_model_training_callbacks(self):
        """Sets up callbacks for AI model training."""
        @self.app.callback(
            [Output('model-training-status', 'children'),
             Output('model-training-history-graph', 'children'),
             Output('model-metrics-display', 'children')],
            [Input('train-model-button', 'n_clicks')],
            [State('model-selection', 'value'),
             State('training-epochs', 'value'),
             State('training-batch-size', 'value'),
             State('analysis-type', 'value'),
             State('url', 'pathname')]
        )
        def handle_train_model(n_clicks, model_name, epochs, batch_size, analysis_type, pathname):
            if not n_clicks or pathname != "/models":
                return dash.no_update, dash.no_update, dash.no_update

            if not model_name or epochs is None or batch_size is None or not analysis_type:
                return dbc.Alert("Please select a model and specify all training parameters.", color="warning"), dash.no_update, dash.no_update

            try:
                payload = {
                    'model_name': model_name, # Or 'agent_type' / 'model_blueprint'
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'analysis_type': analysis_type
                    # Potentially add a 'new_agent_id' or 'new_model_id' for the user to name it
                }
                # API call to start model training. 
                # This endpoint should exist in api_server.py and interact with AIModelManager
                response = requests.post(f"{self.api_base_url}/models/train", json=payload)
                response.raise_for_status()
                data = response.json()

                status_msg = dbc.Alert(data.get("message", "Training initiated successfully."), color="success", className="mt-2")
                
                history_graph = html.Div()
                if 'history' in data and data['history']:
                    # Assuming history is a dict like {'loss': [...], 'val_loss': [...], ...}
                    history_df = pd.DataFrame(data['history'])
                    fig = go.Figure()
                    for col in history_df.columns:
                        if col.lower() != 'epoch': # Assuming 'epoch' is not a plottable metric here or handled differently
                            fig.add_trace(go.Scatter(y=history_df[col], name=col, mode='lines+markers'))
                
                    fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Metric Value", margin=dict(t=30, b=10, l=10, r=10), height=300)
                    history_graph = dcc.Graph(figure=fig)
            
                metrics_display = html.Div([
                    html.H6("Final Metrics:", className="mt-3"),
                    dcc.Markdown(f"```json\n{json.dumps(data.get('metrics', {}), indent=2)}\n```")
                ])
            
                return status_msg, history_graph, metrics_display
            except requests.exceptions.RequestException as e:
                error_detail = e.response.text if e.response else str(e)
                return dbc.Alert(f"API Error during model training: {error_detail}", color="danger"), dash.no_update, dash.no_update
            except Exception as e:
                return dbc.Alert(f"An unexpected error occurred: {e}", color="danger"), dash.no_update, dash.no_update

    def list_existing_models(self, pathname):
        if pathname == "/models":
            try:
                return dbc.Alert("Fetching existing models...", color="info")
            except Exception as e:
                return dbc.Alert(f"Error fetching models: {e}", color="danger")
        return dash.no_update

    def _setup_callbacks(self):
        """Set up all Dash callbacks."""
        # Navigation callback
        @self.app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
        def display_page(pathname):
            return self._render_page_content(pathname)

        # Agent callbacks
        @self.app.callback(
            Output("backtest-settings", "style"),
            [Input("mode-selection", "value")]
        )
        def toggle_backtest_settings(mode):
            if mode == "backtest":
                return {"display": "block"}
            return {"display": "none"}

        # Model training callbacks
        @self.app.callback(
            [Output('model-training-status', 'children'),
             Output('model-training-history-graph', 'children'),
             Output('model-metrics-display', 'children')],
            [Input('train-model-button', 'n_clicks')],
            [State('model-selection', 'value'),
             State('training-epochs', 'value'),
             State('training-batch-size', 'value'),
             State('analysis-type', 'value'),
             State('url', 'pathname')]
        )
        def handle_train_model(n_clicks, model_name, epochs, batch_size, analysis_type, pathname):
            if not n_clicks or pathname != "/models":
                return dash.no_update, dash.no_update, dash.no_update

            if not model_name or epochs is None or batch_size is None or not analysis_type:
                return dbc.Alert("Please select a model and specify all training parameters.", color="warning"), dash.no_update, dash.no_update

            try:
                payload = {
                    'model_name': model_name,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'analysis_type': analysis_type
                }
                response = requests.post(f"{self.api_base_url}/models/train", json=payload)
                response.raise_for_status()
                data = response.json()

                status_msg = dbc.Alert(data.get("message", "Training initiated successfully."), color="success", className="mt-2")
                
                history_graph = html.Div()
                if 'history' in data and data['history']:
                    history_df = pd.DataFrame(data['history'])
                    fig = go.Figure()
                    for col in history_df.columns:
                        if col.lower() != 'epoch':
                            fig.add_trace(go.Scatter(y=history_df[col], name=col, mode='lines+markers'))
                    fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Metric Value", margin=dict(t=30, b=10, l=10, r=10), height=300)
                    history_graph = dcc.Graph(figure=fig)
                
                metrics_display = html.Div([
                    html.H6("Final Metrics:", className="mt-3"),
                    dcc.Markdown(f"```json\n{json.dumps(data.get('metrics', {}), indent=2)}\n```")
                ])
                
                return status_msg, history_graph, metrics_display
            except requests.exceptions.RequestException as e:
                error_detail = e.response.text if e.response else str(e)
                return dbc.Alert(f"API Error during model training: {error_detail}", color="danger"), dash.no_update, dash.no_update
            except Exception as e:
                return dbc.Alert(f"An unexpected error occurred: {e}", color="danger"), dash.no_update, dash.no_update

        # Existing models callback
        @self.app.callback(
            Output('existing-models-list', 'children'),
            [Input('url', 'pathname')] 
        )
        def list_existing_models_callback(pathname):
            return self.list_existing_models(pathname)

    def run(self):
        def list_existing_models(pathname):
            if pathname == "/models":
                try:
                    # response = requests.get(f"{self.api_base_url}/models") # Or /agents if models are part of agents
                    # response.raise_for_status()
                    # models = response.json()
                    # model_list_items = [dbc.ListGroupItem(f"{m.get('name', m.get('id'))} - Type: {m.get('type')}") for m in models]
                    # return dbc.ListGroup(model_list_items)
                    return dbc.Alert("Fetching existing models... (API call pending)", color="info") # Placeholder
                except Exception as e:
                    return dbc.Alert(f"Error fetching models: {e}", color="danger")
            return dash.no_update

    def run(self):
        """Run the Dash app. For production, use a proper WSGI server like Gunicorn or Waitress."""
        try:
            self.app.run_server(debug=True, port=self.dashboard_port, host='0.0.0.0')
        except Exception as e:
            print(f"Error running dashboard: {e}")

# Example usage (for direct script execution, not for production)
if __name__ == '__main__':
    # This config is a placeholder. In a real scenario, load it from a file or environment.
    # Ensure this matches your actual config structure for ports.
    from utils.config_manager import ConfigManager # Assuming ConfigManager is in utils
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
"""
        with open(config_path, 'w') as f:
            f.write(dummy_config_content)
        print(f"Created dummy {config_path} for testing.")

    config_manager = ConfigManager(config_path)
    app_config = config_manager.get_config()

    dashboard = DashDashboard(config=app_config)
    # To run directly for testing (without Gunicorn/Waitress):
    try:
        dashboard.app.run_server(debug=True, port=app_config.get('ui', {}).get('dashboard_port', 8080), host='0.0.0.0')
    except Exception as e:
        print(f"Error running dashboard: {e}")