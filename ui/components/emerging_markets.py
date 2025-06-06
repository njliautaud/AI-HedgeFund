"""
Emerging Markets Component

This module provides the UI components for displaying emerging markets data
in the dashboard.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import pandas as pd
from typing import List, Dict, Any

class EmergingMarketsComponent:
    """Component for displaying emerging markets data."""
    
    def __init__(self):
        """Initialize the component."""
        self.id = 'emerging-markets-component'
    
    def get_layout(self) -> dbc.Container:
        """Get the layout for the emerging markets view."""
        return dbc.Container([
            html.H2("Emerging Markets", className="mb-4"),
            html.Hr(),
            
            # Controls Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Filters", className="bg-light"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Risk Level"),
                                    dcc.Dropdown(
                                        id='risk-level-filter',
                                        options=[
                                            {'label': 'All', 'value': 'all'},
                                            {'label': 'Low', 'value': 'Low'},
                                            {'label': 'Medium', 'value': 'Medium'},
                                            {'label': 'High', 'value': 'High'},
                                        ],
                                        value='all',
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Sort By"),
                                    dcc.Dropdown(
                                        id='sort-by',
                                        options=[
                                            {'label': 'Country', 'value': 'country'},
                                            {'label': 'GDP Growth', 'value': 'gdp_growth'},
                                            {'label': 'Inflation', 'value': 'inflation'},
                                            {'label': 'Risk Level', 'value': 'risk_level'},
                                        ],
                                        value='gdp_growth',
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Sort Order"),
                                    dcc.Dropdown(
                                        id='sort-order',
                                        options=[
                                            {'label': 'Ascending', 'value': 'asc'},
                                            {'label': 'Descending', 'value': 'desc'},
                                        ],
                                        value='desc',
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                ], md=4),
                            ]),
                        ]),
                    ], className="mb-4"),
                ]),
            ]),
            
            # Markets Overview Cards
            dbc.Row([
                dbc.Col([
                    html.Div(id='markets-overview-cards'),
                ]),
            ]),
            
            # Markets Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("Emerging Markets Overview", className="mb-0"),
                        ], className="bg-light"),
                        dbc.CardBody([
                            html.Div(id='markets-table'),
                            dcc.Interval(
                                id='markets-update-interval',
                                interval=300000,  # 5 minutes
                                n_intervals=0
                            ),
                        ]),
                    ]),
                ]),
            ], className="mt-4"),
            
            # Market Details Section
            dbc.Row([
                dbc.Col([
                    html.Div(id='market-details'),
                ]),
            ], className="mt-4"),
            
            # Hidden div to store the markets data
            dcc.Store(id='markets-data-store'),
            
        ], fluid=True, className="py-4")
    
    @staticmethod
    def create_market_card(market: Dict[str, Any]) -> dbc.Card:
        """Create a card for a market overview."""
        # Determine card color based on risk level
        risk_colors = {
            'Low': 'success',
            'Medium': 'warning',
            'High': 'danger'
        }
        
        return dbc.Card(
            [
                dbc.CardHeader([
                    html.H4(market['country'], className="mb-0"),
                    dbc.Badge(
                        market['risk_level'], 
                        color=risk_colors.get(market['risk_level'], 'secondary'),
                        className="float-end"
                    ),
                ], className="d-flex justify-content-between align-items-center"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Small("GDP Growth", className="text-muted"),
                                html.H4(f"{market['gdp_growth']}%", className="mb-0"),
                            ]),
                        ]),
                        dbc.Col([
                            html.Div([
                                html.Small("Inflation", className="text-muted"),
                                html.H4(f"{market['inflation']}%", className="mb-0"),
                            ]),
                        ]),
                        dbc.Col([
                            html.Div([
                                html.Small("Key Industries", className="text-muted"),
                                html.Div(
                                    ", ".join(market['key_industries'][:3]) + 
                                    ("..." if len(market['key_industries']) > 3 else ""),
                                    className="text-truncate"
                                ),
                            ]),
                        ], md=4),
                    ]),
                ]),
                dbc.CardFooter([
                    dbc.Button(
                        "View Details",
                        id={"type": "market-details-button", "index": market['country']},
                        color="primary",
                        size="sm",
                        className="w-100"
                    ),
                ]),
            ],
            className="h-100"
        )
    
    @staticmethod
    def create_market_details(market: Dict[str, Any]) -> dbc.Card:
        """Create a detailed view of a market."""
        return dbc.Card([
            dbc.CardHeader([
                html.H4(f"{market['country']} - Market Details", className="mb-0"),
            ], className="bg-light"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Economic Indicators"),
                        html.Div([
                            html.Div([
                                html.Span("GDP Growth Rate:", className="fw-bold"),
                                html.Span(f" {market['gdp_growth']}%")
                            ], className="mb-2"),
                            html.Div([
                                html.Span("Inflation Rate:", className="fw-bold"),
                                html.Span(f" {market['inflation']}%")
                            ], className="mb-2"),
                            html.Div([
                                html.Span("Risk Level:", className="fw-bold"),
                                dbc.Badge(
                                    market['risk_level'], 
                                    color={
                                        'Low': 'success',
                                        'Medium': 'warning',
                                        'High': 'danger'
                                    }.get(market['risk_level'], 'secondary'),
                                    className="ms-2"
                                )
                            ], className="mb-3"),
                        ]),
                        
                        html.H5("Key Industries", className="mt-4"),
                        html.Ul([html.Li(industry) for industry in market['key_industries']], className="mb-0"),
                        
                        html.H5("Market ETFs", className="mt-4"),
                        html.Div([
                            dbc.Badge(etf, color="info", className="me-2 mb-2") 
                            for etf in market['etf_tickers']
                        ], className="mb-3"),
                    ], md=6),
                    
                    dbc.Col([
                        html.H5("Major Companies"),
                        dash_table.DataTable(
                            id='companies-table',
                            columns=[
                                {"name": "Name", "id": "name"},
                                {"name": "Ticker", "id": "ticker"},
                                {"name": "Sector", "id": "sector"},
                            ],
                            data=market['major_companies'],
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '8px',
                                'minWidth': '100px',
                                'maxWidth': '300px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                            },
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                }
                            ],
                            page_size=10,
                        ),
                    ], md=6),
                ]),
            ]),
        ], className="mb-4")
