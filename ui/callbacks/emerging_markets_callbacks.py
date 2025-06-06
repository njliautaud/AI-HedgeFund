"""
Callbacks for the Emerging Markets component.
"""

from dash import Input, Output, State, html, dcc, callback_context, no_update
import dash_bootstrap_components as dbc
import dash_table
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime

from src.data.emerging_markets import EmergingMarketsData
from ui.components.emerging_markets import EmergingMarketsComponent

# Initialize the data fetcher
em_data = EmergingMarketsData()

# Initialize the component
em_component = EmergingMarketsComponent()

def register_callbacks(app):
    """Register all callbacks for the emerging markets component."""
    
    @app.callback(
        Output('markets-data-store', 'data'),
        [Input('markets-update-interval', 'n_intervals')]
    )
    def update_markets_data(n_intervals):
        """Update the markets data at regular intervals."""
        markets = em_data.get_all_markets()
        
        # Convert the markets to a list of dictionaries for JSON serialization
        markets_data = []
        for market in markets:
            market_dict = {
                'country': market.country,
                'gdp_growth': market.gdp_growth,
                'inflation': market.inflation,
                'risk_level': market.risk_level,
                'key_industries': market.key_industries,
                'etf_tickers': market.etf_tickers,
                'major_companies': [
                    {
                        'name': company['name'],
                        'ticker': company['ticker'],
                        'sector': company['sector']
                    }
                    for company in market.major_companies
                ]
            }
            markets_data.append(market_dict)
        
        return markets_data
    
    @app.callback(
        [
            Output('markets-overview-cards', 'children'),
            Output('markets-table', 'children')
        ],
        [
            Input('markets-data-store', 'data'),
            Input('risk-level-filter', 'value'),
            Input('sort-by', 'value'),
            Input('sort-order', 'value')
        ]
    )
    def update_markets_view(
        markets_data, 
        risk_filter, 
        sort_by, 
        sort_order
    ) -> Tuple[dbc.Row, dash_table.DataTable]:
        """Update the markets overview and table based on filters and sorting."""
        if not markets_data:
            return no_update, no_update
        
        # Apply filters
        if risk_filter != 'all':
            filtered_markets = [
                m for m in markets_data 
                if m['risk_level'] == risk_filter
            ]
        else:
            filtered_markets = markets_data
        
        # Apply sorting
        reverse_sort = (sort_order == 'desc')
        
        if sort_by == 'gdp_growth':
            filtered_markets.sort(key=lambda x: x['gdp_growth'], reverse=reverse_sort)
        elif sort_by == 'inflation':
            filtered_markets.sort(key=lambda x: x['inflation'], reverse=reverse_sort)
        elif sort_by == 'risk_level':
            # Map risk levels to numerical values for sorting
            risk_levels = {'High': 2, 'Medium': 1, 'Low': 0}
            filtered_markets.sort(
                key=lambda x: risk_levels.get(x['risk_level'], 0), 
                reverse=reverse_sort
            )
        else:  # Default sort by country
            filtered_markets.sort(key=lambda x: x['country'], reverse=reverse_sort)
        
        # Create market cards
        cards = []
        for market in filtered_markets:
            cards.append(
                dbc.Col(
                    em_component.create_market_card(market),
                    className="mb-4",
                    md=6,
                    lg=4
                )
            )
        
        # Create table data
        table_data = [
            {
                'Country': m['country'],
                'GDP Growth (%)': m['gdp_growth'],
                'Inflation (%)': m['inflation'],
                'Risk Level': m['risk_level'],
                'Key Industries': ', '.join(m['key_industries'][:3]) + 
                                ('...' if len(m['key_industries']) > 3 else '')
            }
            for m in filtered_markets
        ]
        
        # Create the table
        table = dash_table.DataTable(
            id='markets-datatable',
            columns=[
                {"name": "Country", "id": "Country"},
                {"name": "GDP Growth (%)", "id": "GDP Growth (%)"},
                {"name": "Inflation (%)", "id": "Inflation (%)"},
                {"name": "Risk Level", "id": "Risk Level"},
                {"name": "Key Industries", "id": "Key Industries"},
            ],
            data=table_data,
            style_table={
                'overflowX': 'auto',
                'width': '100%',
                'minWidth': '100%',
            },
            style_cell={
                'textAlign': 'left',
                'padding': '12px',
                'minWidth': '100px',
                'maxWidth': '300px',
                'whiteSpace': 'normal',
                'height': 'auto',
                'border': '1px solid #dee2e6',
            },
            style_header={
                'backgroundColor': '#f8f9fa',
                'fontWeight': 'bold',
                'border': '1px solid #dee2e6',
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgba(0, 0, 0, 0.03)'
                },
                {
                    'if': {
                        'filter_query': '{Risk Level} = "High"',
                        'column_id': 'Risk Level'
                    },
                    'color': '#dc3545',
                    'fontWeight': 'bold'
                },
                {
                    'if': {
                        'filter_query': '{Risk Level} = "Medium"',
                        'column_id': 'Risk Level'
                    },
                    'color': '#ffc107',
                    'fontWeight': 'bold'
                },
                {
                    'if': {
                        'filter_query': '{Risk Level} = "Low"',
                        'column_id': 'Risk Level'
                    },
                    'color': '#198754',
                    'fontWeight': 'bold'
                },
            ],
            page_size=10,
            sort_action='native',
            filter_action='native',
            row_selectable='single',
            selected_rows=[],
            style_cell_conditional=[
                {'if': {'column_id': 'Country'}, 'width': '15%'},
                {'if': {'column_id': 'GDP Growth (%)'}, 'width': '15%'},
                {'if': {'column_id': 'Inflation (%)'}, 'width': '15%'},
                {'if': {'column_id': 'Risk Level'}, 'width': '15%'},
                {'if': {'column_id': 'Key Industries'}, 'width': '40%'},
            ],
        )
        
        return dbc.Row(cards, className="g-4"), table
    
    @app.callback(
        Output('market-details', 'children'),
        [Input('markets-datatable', 'selected_rows')],
        [State('markets-data-store', 'data')]
    )
    def show_market_details(selected_rows, markets_data):
        """Show details for the selected market."""
        if not selected_rows or not markets_data:
            return None
        
        # Get the selected market
        selected_market = markets_data[selected_rows[0]]
        
        # Create and return the market details card
        return em_component.create_market_details(selected_market)
    
    # Handle market details button clicks
    @app.callback(
        Output('market-details', 'children', allow_duplicate=True),
        [
            Input({'type': 'market-details-button', 'index': 'ALL'}, 'n_clicks')
        ],
        [
            State('markets-data-store', 'data')
        ],
        prevent_initial_call=True
    )
    def handle_market_details_button(_, markets_data):
        """Handle clicks on market details buttons."""
        ctx = callback_context
        
        if not ctx.triggered or not markets_data:
            return None
        
        # Get the button ID that was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        try:
            # Parse the button ID to get the market name
            button_data = json.loads(button_id.replace("'", '"'))
            market_name = button_data['index']
            
            # Find the market data
            for market in markets_data:
                if market['country'] == market_name:
                    return em_component.create_market_details(market)
            
            return None
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing button click: {e}")
            return None
    
    return app
