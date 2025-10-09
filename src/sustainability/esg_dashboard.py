"""
ESG Metrics Dashboard using Plotly and Dash.

This module implements a real-time monitoring dashboard for ESG metrics
with trend analysis, comparative visualizations, and alerting capabilities.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings

# Dashboard dependencies
try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    warnings.warn("Dash/Plotly not available. Install with: pip install dash plotly")
    
    # Create mock objects when Dash is not available
    class MockHtml:
        @staticmethod
        def Div(*args, **kwargs):
            return "MockDiv"
        @staticmethod
        def H1(*args, **kwargs):
            return "MockH1"
        @staticmethod
        def H2(*args, **kwargs):
            return "MockH2"
        @staticmethod
        def H3(*args, **kwargs):
            return "MockH3"
        @staticmethod
        def Label(*args, **kwargs):
            return "MockLabel"
        @staticmethod
        def Button(*args, **kwargs):
            return "MockButton"
        @staticmethod
        def Span(*args, **kwargs):
            return "MockSpan"
        @staticmethod
        def Strong(*args, **kwargs):
            return "MockStrong"
    
    class MockGo:
        @staticmethod
        def Figure(*args, **kwargs):
            return "MockFigure"
        @staticmethod
        def Scatter(*args, **kwargs):
            return "MockScatter"
        @staticmethod
        def Bar(*args, **kwargs):
            return "MockBar"
        @staticmethod
        def Pie(*args, **kwargs):
            return "MockPie"
    
    html = MockHtml()
    go = MockGo()

try:
    from ..core.logging import get_logger
    from .esg_metrics import ESGMetricsCollector, ESGMetric, ESGScore, ESGReport, ESGCategory, ESGMetricType
    from .carbon_calculator import CarbonCalculator
    from .energy_tracker import EnergyTracker
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.logging import get_logger
    from sustainability.esg_metrics import ESGMetricsCollector, ESGMetric, ESGScore, ESGReport, ESGCategory, ESGMetricType
    from sustainability.carbon_calculator import CarbonCalculator
    from sustainability.energy_tracker import EnergyTracker

logger = get_logger(__name__)


class ESGDashboard:
    """Real-time ESG metrics dashboard with Plotly/Dash."""
    
    def __init__(self, esg_collector: Optional[ESGMetricsCollector] = None,
                 port: int = 8050, debug: bool = False):
        
        if not DASH_AVAILABLE:
            raise ImportError("Dash and Plotly are required for ESG Dashboard. "
                            "Install with: pip install dash plotly")
        
        self.esg_collector = esg_collector or ESGMetricsCollector()
        self.port = port
        self.debug = debug
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.app.title = "ESG Metrics Dashboard"
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info(f"ESG Dashboard initialized on port {port}")
    
    def _setup_layout(self):
        """Setup the dashboard layout."""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("ESG Metrics Dashboard", 
                       className="dashboard-title",
                       style={'textAlign': 'center', 'color': '#2E8B57', 'marginBottom': '30px'}),
                
                html.Div([
                    html.Div([
                        html.Label("Time Period:"),
                        dcc.Dropdown(
                            id='time-period-dropdown',
                            options=[
                                {'label': 'Last 24 Hours', 'value': 1},
                                {'label': 'Last 7 Days', 'value': 7},
                                {'label': 'Last 30 Days', 'value': 30},
                                {'label': 'Last 90 Days', 'value': 90}
                            ],
                            value=7,
                            style={'width': '200px'}
                        )
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Button('Refresh Data', id='refresh-button', 
                                  style={'backgroundColor': '#2E8B57', 'color': 'white', 
                                        'border': 'none', 'padding': '10px 20px', 
                                        'borderRadius': '5px', 'cursor': 'pointer'})
                    ], style={'display': 'inline-block'})
                ], style={'textAlign': 'center', 'marginBottom': '30px'})
            ]),
            
            # Alert Section
            html.Div(id='alerts-section', style={'marginBottom': '20px'}),
            
            # ESG Score Cards
            html.Div([
                html.H2("ESG Scores", style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div(id='esg-score-cards', style={'display': 'flex', 'justifyContent': 'space-around'})
            ], style={'marginBottom': '40px'}),
            
            # Main Charts Section
            html.Div([
                # ESG Trends Chart
                html.Div([
                    dcc.Graph(id='esg-trends-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # Category Breakdown Chart
                html.Div([
                    dcc.Graph(id='category-breakdown-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Detailed Metrics Section
            html.Div([
                html.H2("Detailed Metrics", style={'textAlign': 'center', 'marginBottom': '20px'}),
                
                # Environmental Metrics
                html.Div([
                    html.H3("Environmental Metrics", style={'color': '#228B22'}),
                    dcc.Graph(id='environmental-metrics-chart')
                ], style={'marginBottom': '30px'}),
                
                # Social Metrics
                html.Div([
                    html.H3("Social Metrics", style={'color': '#4169E1'}),
                    dcc.Graph(id='social-metrics-chart')
                ], style={'marginBottom': '30px'}),
                
                # Governance Metrics
                html.Div([
                    html.H3("Governance Metrics", style={'color': '#8B4513'}),
                    dcc.Graph(id='governance-metrics-chart')
                ], style={'marginBottom': '30px'})
            ]),
            
            # Benchmarking Section
            html.Div([
                html.H2("Benchmarking", style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div([
                    dcc.Graph(id='benchmark-comparison-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='target-progress-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Recommendations Section
            html.Div([
                html.H2("Recommendations", style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div(id='recommendations-section')
            ], style={'marginTop': '40px'}),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            ),
            
            # Hidden div to store data
            html.Div(id='hidden-data-store', style={'display': 'none'})
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""
        
        @self.app.callback(
            [Output('hidden-data-store', 'children'),
             Output('alerts-section', 'children'),
             Output('esg-score-cards', 'children'),
             Output('esg-trends-chart', 'figure'),
             Output('category-breakdown-chart', 'figure'),
             Output('environmental-metrics-chart', 'figure'),
             Output('social-metrics-chart', 'figure'),
             Output('governance-metrics-chart', 'figure'),
             Output('benchmark-comparison-chart', 'figure'),
             Output('target-progress-chart', 'figure'),
             Output('recommendations-section', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks'),
             Input('time-period-dropdown', 'value')]
        )
        def update_dashboard(n_intervals, n_clicks, time_period):
            """Update all dashboard components."""
            
            try:
                # Get metrics data
                metrics = self._get_sample_metrics_data(time_period)
                
                if not metrics:
                    # Return empty components if no data
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(text="No data available", 
                                           xref="paper", yref="paper",
                                           x=0.5, y=0.5, showarrow=False)
                    
                    return (
                        json.dumps([]),  # hidden-data-store
                        html.Div("No alerts"),  # alerts-section
                        html.Div("No score data"),  # esg-score-cards
                        empty_fig,  # esg-trends-chart
                        empty_fig,  # category-breakdown-chart
                        empty_fig,  # environmental-metrics-chart
                        empty_fig,  # social-metrics-chart
                        empty_fig,  # governance-metrics-chart
                        empty_fig,  # benchmark-comparison-chart
                        empty_fig,  # target-progress-chart
                        html.Div("No recommendations")  # recommendations-section
                    )
                
                # Calculate ESG score
                esg_score = self.esg_collector.calculate_esg_score(metrics)
                
                # Generate alerts and recommendations
                alerts = self.esg_collector.generate_alerts(metrics)
                recommendations = self.esg_collector.generate_recommendations(metrics)
                
                # Create components
                alerts_component = self._create_alerts_component(alerts)
                score_cards = self._create_score_cards(esg_score)
                trends_chart = self._create_trends_chart(metrics)
                breakdown_chart = self._create_category_breakdown_chart(esg_score)
                env_chart = self._create_metrics_chart(metrics, ESGCategory.ENVIRONMENTAL)
                social_chart = self._create_metrics_chart(metrics, ESGCategory.SOCIAL)
                gov_chart = self._create_metrics_chart(metrics, ESGCategory.GOVERNANCE)
                benchmark_chart = self._create_benchmark_chart(metrics)
                target_chart = self._create_target_progress_chart(metrics)
                recommendations_component = self._create_recommendations_component(recommendations)
                
                # Store data for other callbacks
                data_store = json.dumps([metric.to_dict() for metric in metrics])
                
                return (
                    data_store,
                    alerts_component,
                    score_cards,
                    trends_chart,
                    breakdown_chart,
                    env_chart,
                    social_chart,
                    gov_chart,
                    benchmark_chart,
                    target_chart,
                    recommendations_component
                )
                
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                
                # Return error components
                error_fig = go.Figure()
                error_fig.add_annotation(text=f"Error: {str(e)}", 
                                       xref="paper", yref="paper",
                                       x=0.5, y=0.5, showarrow=False)
                
                return (
                    json.dumps([]),
                    html.Div(f"Error: {str(e)}", style={'color': 'red'}),
                    html.Div("Error loading scores"),
                    error_fig, error_fig, error_fig, error_fig, error_fig, error_fig, error_fig,
                    html.Div("Error loading recommendations")
                )
    
    def _get_sample_metrics_data(self, days: int) -> List[ESGMetric]:
        """Get sample ESG metrics data for demonstration."""
        
        # In a real implementation, this would fetch actual data
        # For now, generate sample data
        
        from .energy_tracker import EnergyReport
        from .carbon_calculator import CarbonFootprint
        
        # Create sample energy reports
        energy_reports = []
        carbon_footprints = []
        
        for i in range(10):  # Sample 10 experiments
            # Sample energy report
            energy_report = EnergyReport(
                experiment_id=f"exp_{i}",
                start_time=datetime.now() - timedelta(days=days-i),
                end_time=datetime.now() - timedelta(days=days-i, hours=-1),
                duration_seconds=3600,
                total_energy_kwh=0.05 + np.random.normal(0, 0.01),
                cpu_energy_kwh=0.035 + np.random.normal(0, 0.005),
                gpu_energy_kwh=0.015 + np.random.normal(0, 0.005)
            )
            energy_reports.append(energy_report)
            
            # Sample carbon footprint
            carbon_footprint = CarbonFootprint(
                experiment_id=f"exp_{i}",
                timestamp=datetime.now() - timedelta(days=days-i),
                energy_kwh=energy_report.total_energy_kwh,
                operational_emissions_kg=energy_report.total_energy_kwh * 0.4,
                embodied_emissions_kg=energy_report.total_energy_kwh * 0.04,
                total_emissions_kg=energy_report.total_energy_kwh * 0.44,
                region="US",
                carbon_intensity_gco2_kwh=400 + np.random.normal(0, 50)
            )
            carbon_footprints.append(carbon_footprint)
        
        # Collect metrics
        return self.esg_collector.collect_all_metrics(
            energy_reports=energy_reports,
            carbon_footprints=carbon_footprints,
            fairness_scores={'demographic_parity': 0.85, 'equal_opportunity': 0.88},
            privacy_metrics={'overall_privacy_score': 0.9},
            compliance_data={'overall_compliance': 0.95}
        )
    
    def _create_alerts_component(self, alerts: List[str]) -> html.Div:
        """Create alerts component."""
        
        if not alerts:
            return html.Div([
                html.Div("✅ All ESG metrics are within acceptable ranges", 
                        style={'color': 'green', 'fontWeight': 'bold', 'textAlign': 'center'})
            ])
        
        alert_items = []
        for alert in alerts:
            if "CRITICAL" in alert:
                color = 'red'
                icon = "🚨"
            elif "WARNING" in alert:
                color = 'orange'
                icon = "⚠️"
            else:
                color = 'blue'
                icon = "ℹ️"
            
            alert_items.append(
                html.Div([
                    html.Span(icon, style={'marginRight': '10px'}),
                    html.Span(alert)
                ], style={'color': color, 'marginBottom': '10px', 'padding': '10px',
                         'border': f'1px solid {color}', 'borderRadius': '5px'})
            )
        
        return html.Div(alert_items)
    
    def _create_score_cards(self, esg_score: ESGScore) -> List[html.Div]:
        """Create ESG score cards."""
        
        def create_score_card(title: str, score: float, color: str) -> html.Div:
            return html.Div([
                html.H3(title, style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"{score:.1f}", 
                        style={'fontSize': '48px', 'fontWeight': 'bold', 
                              'textAlign': 'center', 'color': color}),
                html.Div("/ 100", style={'textAlign': 'center', 'color': 'gray'})
            ], style={'border': f'2px solid {color}', 'borderRadius': '10px', 
                     'padding': '20px', 'width': '200px', 'textAlign': 'center'})
        
        return [
            create_score_card("Environmental", esg_score.environmental_score, '#228B22'),
            create_score_card("Social", esg_score.social_score, '#4169E1'),
            create_score_card("Governance", esg_score.governance_score, '#8B4513'),
            create_score_card("Overall ESG", esg_score.overall_score, '#2E8B57')
        ]
    
    def _create_trends_chart(self, metrics: List[ESGMetric]) -> go.Figure:
        """Create ESG trends chart."""
        
        # Group metrics by category and time
        df_data = []
        for metric in metrics:
            df_data.append({
                'timestamp': metric.timestamp,
                'category': metric.category.value,
                'metric_type': metric.metric_type.value,
                'value': metric.value,
                'performance_ratio': metric.performance_ratio() or 0
            })
        
        if not df_data:
            fig = go.Figure()
            fig.add_annotation(text="No trend data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        df = pd.DataFrame(df_data)
        
        # Create trends by category
        fig = go.Figure()
        
        categories = df['category'].unique()
        colors = {'environmental': '#228B22', 'social': '#4169E1', 'governance': '#8B4513'}
        
        for category in categories:
            cat_data = df[df['category'] == category]
            # Average performance ratio over time
            avg_performance = cat_data.groupby('timestamp')['performance_ratio'].mean().reset_index()
            
            fig.add_trace(go.Scatter(
                x=avg_performance['timestamp'],
                y=avg_performance['performance_ratio'] * 100,  # Convert to percentage
                mode='lines+markers',
                name=category.title(),
                line=dict(color=colors.get(category, '#000000'))
            ))
        
        fig.update_layout(
            title="ESG Performance Trends",
            xaxis_title="Time",
            yaxis_title="Performance Score (%)",
            hovermode='x unified'
        )
        
        return fig
    
    def _create_category_breakdown_chart(self, esg_score: ESGScore) -> go.Figure:
        """Create category breakdown pie chart."""
        
        labels = ['Environmental', 'Social', 'Governance']
        values = [esg_score.environmental_score, esg_score.social_score, esg_score.governance_score]
        colors = ['#228B22', '#4169E1', '#8B4513']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{value:.1f}/100<br>(%{percent})'
        )])
        
        fig.update_layout(
            title="ESG Score Breakdown",
            showlegend=True
        )
        
        return fig
    
    def _create_metrics_chart(self, metrics: List[ESGMetric], category: ESGCategory) -> go.Figure:
        """Create detailed metrics chart for a specific category."""
        
        category_metrics = [m for m in metrics if m.category == category]
        
        if not category_metrics:
            fig = go.Figure()
            fig.add_annotation(text=f"No {category.value} metrics available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create bar chart comparing current vs target values
        metric_names = [m.metric_type.value.replace('_', ' ').title() for m in category_metrics]
        current_values = [m.value for m in category_metrics]
        target_values = [m.target_value or 0 for m in category_metrics]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current',
            x=metric_names,
            y=current_values,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Target',
            x=metric_names,
            y=target_values,
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title=f"{category.value.title()} Metrics: Current vs Target",
            xaxis_title="Metrics",
            yaxis_title="Value",
            barmode='group',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_benchmark_chart(self, metrics: List[ESGMetric]) -> go.Figure:
        """Create benchmark comparison chart."""
        
        # Compare current performance vs industry benchmarks
        metric_names = []
        current_values = []
        benchmark_values = []
        
        for metric in metrics:
            if metric.benchmark_value is not None:
                metric_names.append(metric.metric_type.value.replace('_', ' ').title())
                current_values.append(metric.value)
                benchmark_values.append(metric.benchmark_value)
        
        if not metric_names:
            fig = go.Figure()
            fig.add_annotation(text="No benchmark data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Our Performance',
            x=metric_names,
            y=current_values,
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            name='Industry Benchmark',
            x=metric_names,
            y=benchmark_values,
            marker_color='gray'
        ))
        
        fig.update_layout(
            title="Performance vs Industry Benchmarks",
            xaxis_title="Metrics",
            yaxis_title="Value",
            barmode='group',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_target_progress_chart(self, metrics: List[ESGMetric]) -> go.Figure:
        """Create target progress chart."""
        
        # Show progress towards targets as percentage
        metric_names = []
        progress_percentages = []
        
        for metric in metrics:
            performance_ratio = metric.performance_ratio()
            if performance_ratio is not None:
                metric_names.append(metric.metric_type.value.replace('_', ' ').title())
                progress_percentages.append(min(performance_ratio * 100, 100))
        
        if not metric_names:
            fig = go.Figure()
            fig.add_annotation(text="No target progress data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Color code based on progress
        colors = []
        for progress in progress_percentages:
            if progress >= 100:
                colors.append('green')
            elif progress >= 80:
                colors.append('yellow')
            else:
                colors.append('red')
        
        fig = go.Figure(data=[go.Bar(
            x=metric_names,
            y=progress_percentages,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in progress_percentages],
            textposition='auto'
        )])
        
        # Add target line at 100%
        fig.add_hline(y=100, line_dash="dash", line_color="blue", 
                     annotation_text="Target (100%)")
        
        fig.update_layout(
            title="Progress Towards ESG Targets",
            xaxis_title="Metrics",
            yaxis_title="Progress (%)",
            xaxis_tickangle=-45,
            yaxis_range=[0, 120]
        )
        
        return fig
    
    def _create_recommendations_component(self, recommendations: List[str]) -> html.Div:
        """Create recommendations component."""
        
        if not recommendations:
            return html.Div("No recommendations available.")
        
        recommendation_items = []
        for i, rec in enumerate(recommendations, 1):
            recommendation_items.append(
                html.Div([
                    html.Strong(f"{i}. "),
                    html.Span(rec)
                ], style={'marginBottom': '15px', 'padding': '10px',
                         'backgroundColor': '#f0f8ff', 'borderRadius': '5px',
                         'border': '1px solid #e0e0e0'})
            )
        
        return html.Div(recommendation_items)
    
    def run(self, host: str = '127.0.0.1'):
        """Run the dashboard server."""
        
        if not DASH_AVAILABLE:
            raise ImportError("Dash is required to run the dashboard")
        
        logger.info(f"Starting ESG Dashboard on http://{host}:{self.port}")
        self.app.run_server(host=host, port=self.port, debug=self.debug)
    
    def get_app(self):
        """Get the Dash app instance for external deployment."""
        return self.app


# Utility functions for easy dashboard deployment

def create_esg_dashboard(esg_collector: Optional[ESGMetricsCollector] = None,
                        port: int = 8050, debug: bool = False) -> ESGDashboard:
    """
    Create and configure ESG dashboard.
    
    Args:
        esg_collector: ESG metrics collector instance
        port: Port to run dashboard on
        debug: Enable debug mode
        
    Returns:
        Configured ESG dashboard instance
    """
    return ESGDashboard(esg_collector=esg_collector, port=port, debug=debug)


def run_esg_dashboard(esg_collector: Optional[ESGMetricsCollector] = None,
                     host: str = '127.0.0.1', port: int = 8050, debug: bool = False):
    """
    Run ESG dashboard server.
    
    Args:
        esg_collector: ESG metrics collector instance
        host: Host to bind to
        port: Port to run on
        debug: Enable debug mode
    """
    dashboard = create_esg_dashboard(esg_collector, port, debug)
    dashboard.run(host)


if __name__ == "__main__":
    # Run dashboard with sample data
    if DASH_AVAILABLE:
        run_esg_dashboard(debug=True)
    else:
        print("Dash/Plotly not available. Install with: pip install dash plotly")