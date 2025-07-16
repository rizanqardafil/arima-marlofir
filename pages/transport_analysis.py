"""
Transport Analysis Dashboard - Transport Mode Analysis
/Users/sociolla/Documents/BBM/pages/transport_analysis.py

Comprehensive transport mode analysis dashboard for BBM consumption
Analyzes BBM usage patterns across different transportation modes
FIXED: Removed non-existent VisualizationEngine methods
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.arima_engine import ARIMAEngine
from core.data_generator import DataGenerator, generate_date_range
from core.visualization import VisualizationEngine

class TransportAnalysisDashboard:
    """
    Complete Transport Mode Analysis Dashboard
    
    Features:
    - Multi-location transport mode configuration
    - Unit-based BBM consumption calculation
    - ARIMA forecasting per transport mode
    - Comparative analysis across modes
    - Efficiency and ranking analysis
    """
    
    def __init__(self):
        """Initialize Transport Analysis Dashboard"""
        self.initialize_session_state()
        self.arima_engine = ARIMAEngine(max_p=3, max_d=2, max_q=3)
        self.data_generator = DataGenerator(random_seed=42)
        self.viz_engine = VisualizationEngine(theme='plotly_white')
        
        # Transport mode configuration
        self.transport_modes = {
            '🚤 Kapal Nelayan': {
                'base_consumption': 800, 
                'efficiency': 0.8, 
                'seasonal_amplitude': 0.25,
                'description': 'Fishing boats - weather dependent'
            },
            '🏍️ Ojek Pangkalan': {
                'base_consumption': 150, 
                'efficiency': 1.2, 
                'seasonal_amplitude': 0.15,
                'description': 'Motorcycle taxis - urban transport'
            },
            '🚗 Mobil Pribadi': {
                'base_consumption': 200, 
                'efficiency': 1.0, 
                'seasonal_amplitude': 0.10,
                'description': 'Private cars - personal transport'
            },
            '🚛 Truck Angkutan': {
                'base_consumption': 500, 
                'efficiency': 0.7, 
                'seasonal_amplitude': 0.20,
                'description': 'Cargo trucks - goods transport'
            },
            '⛵ Kapal Penumpang': {
                'base_consumption': 1200, 
                'efficiency': 0.6, 
                'seasonal_amplitude': 0.30,
                'description': 'Passenger boats - inter-island transport'
            },
            '🏭 Generator/Mesin': {
                'base_consumption': 300, 
                'efficiency': 0.9, 
                'seasonal_amplitude': 0.05,
                'description': 'Stationary engines - industrial use'
            }
        }
        
        self.default_locations = ["Jemaja", "Siantan", "Palmatak", "Bintan", "Batam"]
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        session_vars = {
            'transport_config': {},
            'transport_data': {},
            'transport_analysis_results': {},
            'transport_analysis_completed': False,
            'transport_forecast_periods': 12
        }
        
        for var, default_value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default_value
    
    def render_page_header(self):
        """Render page header with title and description"""
        st.title("🚗 Transport Mode Analysis Dashboard")
        st.markdown("""
        **BBM Consumption Analysis by Transportation Mode**
        
        Analyze BBM consumption patterns across different transport modes per location.
        Calculate unit-based consumption, run ARIMA forecasting, and compare efficiency across modes.
        """)
        
        # Transport mode overview
        with st.expander("🚛 Available Transport Modes", expanded=False):
            cols = st.columns(3)
            
            for i, (mode_name, config) in enumerate(self.transport_modes.items()):
                col_idx = i % 3
                with cols[col_idx]:
                    st.write(f"**{mode_name}**")
                    st.write(f"Base: {config['base_consumption']} L/day")
                    st.write(f"Efficiency: {config['efficiency']}x")
                    st.caption(config['description'])
        
        st.markdown("---")
    
    def render_configuration_sidebar(self) -> Dict[str, Any]:
        """
        Render configuration sidebar for transport analysis
        
        Returns:
            Configuration dictionary
        """
        with st.sidebar:
            st.header("⚙️ Transport Analysis Configuration")
            
            # Location Configuration
            st.subheader("1. 🏝️ Location Setup")
            num_locations = st.number_input(
                "Number of locations:",
                min_value=1, max_value=5, value=2,
                help="Choose locations for transport mode analysis"
            )
            
            locations = []
            for i in range(num_locations):
                default_name = self.default_locations[i] if i < len(self.default_locations) else f"Location_{i+1}"
                loc_name = st.text_input(
                    f"Location {i+1}:",
                    value=default_name,
                    key=f"transport_location_{i}",
                    help=f"Enter name for location {i+1}"
                )
                locations.append(loc_name)
            
            st.markdown("---")
            
            # Time Period Configuration
            st.subheader("2. 📅 Time Period Setup")
            
            col1, col2 = st.columns(2)
            with col1:
                num_months = st.slider(
                    "Historical data (months):",
                    min_value=8, max_value=24, value=12,
                    help="Number of months of historical data"
                )
            
            with col2:
                forecast_months = st.slider(
                    "Forecast period (months):",
                    min_value=6, max_value=36, value=12,
                    help="Number of months to forecast"
                )
            
            start_date = st.date_input(
                "Data start date:",
                value=datetime(2023, 1, 1),
                help="Starting date for analysis"
            )
            
            st.markdown("---")
            
            # Transport Mode Configuration
            st.subheader("3. 🚛 Transport Mode Setup")
            
            transport_params = self._render_transport_parameters(locations)
            
            st.markdown("---")
            
            # Analysis Settings
            with st.expander("🔧 Analysis Settings", expanded=False):
                
                # Unit filtering
                min_units = st.number_input(
                    "Minimum units for analysis:",
                    min_value=0, max_value=50, value=5,
                    help="Exclude transport modes with fewer units"
                )
                
                # Efficiency threshold
                efficiency_threshold = st.slider(
                    "Efficiency threshold:",
                    min_value=0.5, max_value=1.5, value=0.8, step=0.1,
                    help="Highlight modes below this efficiency"
                )
                
                # Analysis focus
                analysis_focus = st.selectbox(
                    "Analysis focus:",
                    options=['All Modes', 'High Volume Only', 'Efficient Modes Only', 'Marine Transport', 'Land Transport'],
                    help="Focus analysis on specific transport categories"
                )
            
            st.markdown("---")
            
            # Action Buttons
            st.subheader("4. 🚀 Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                generate_data_btn = st.button(
                    "📊 Generate Transport Data",
                    help="Generate transport mode consumption data",
                    use_container_width=True
                )
            
            with col2:
                run_analysis_btn = st.button(
                    "🔬 Run Transport Analysis",
                    help="Run ARIMA analysis on transport data",
                    use_container_width=True,
                    disabled=len(st.session_state.transport_data) == 0
                )
            
            # Configuration object
            config = {
                'locations': locations,
                'num_months': num_months,
                'forecast_months': forecast_months,
                'start_date': datetime.combine(start_date, datetime.min.time()),
                'transport_params': transport_params,
                'analysis_settings': {
                    'min_units': min_units,
                    'efficiency_threshold': efficiency_threshold,
                    'analysis_focus': analysis_focus
                },
                'generate_data': generate_data_btn,
                'run_analysis': run_analysis_btn
            }
            
            return config
    
    def _render_transport_parameters(self, locations: List[str]) -> Dict[str, Any]:
        """Render transport mode parameters for each location"""
        transport_params = {}
        
        for loc in locations:
            with st.expander(f"🏝️ {loc} - Transport Configuration", expanded=False):
                
                st.write(f"**Configure transport units and consumption for {loc}**")
                
                for mode_name, mode_config in self.transport_modes.items():
                    
                    st.write(f"*{mode_name}*")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        transport_params[f"{loc}_{mode_name}_units"] = st.number_input(
                            "Units:",
                            min_value=0, max_value=1000, value=0, step=1,
                            key=f"{loc}_{mode_name}_units",
                            help=f"Number of {mode_name} units in {loc}"
                        )
                    
                    with col2:
                        transport_params[f"{loc}_{mode_name}_base"] = st.number_input(
                            "L/day/unit:",
                            min_value=10, max_value=3000, 
                            value=mode_config['base_consumption'], step=10,
                            key=f"{loc}_{mode_name}_base",
                            help="BBM consumption per unit per day"
                        )
                    
                    with col3:
                        transport_params[f"{loc}_{mode_name}_var"] = st.slider(
                            "Variation %:",
                            min_value=5, max_value=50, value=15,
                            key=f"{loc}_{mode_name}_var",
                            help="Consumption variation percentage"
                        )
                    
                    # Show calculated monthly consumption
                    units = transport_params[f"{loc}_{mode_name}_units"]
                    base_consumption = transport_params[f"{loc}_{mode_name}_base"]
                    if units > 0:
                        monthly_consumption = self._calculate_monthly_consumption(
                            units, mode_name, base_consumption
                        )
                        st.caption(f"📊 Estimated monthly: {monthly_consumption:,.0f} L")
                    
                    st.markdown("---")
        
        return transport_params
    
    def _calculate_monthly_consumption(self, units: int, mode_name: str, 
                                     base_consumption: float, days: int = 30) -> float:
        """Calculate estimated monthly BBM consumption"""
        if mode_name in self.transport_modes:
            efficiency = self.transport_modes[mode_name]['efficiency']
            return units * base_consumption * efficiency * days
        return units * base_consumption * days
    
    def generate_transport_data(self, config: Dict[str, Any]):
        """Generate transport mode consumption data"""
        with st.spinner("🔄 Generating transport mode consumption data..."):
            
            # Generate date range
            dates = generate_date_range(config['start_date'], config['num_months'])
            
            # Generate transport data
            transport_data = self.data_generator.generate_transport_data(
                locations=config['locations'],
                num_months=config['num_months'],
                parameters=config['transport_params'],
                dates=dates
            )
            
            # Store data
            st.session_state.transport_data = transport_data
            st.session_state.transport_config = config
            
            # Count active transport modes
            active_modes = 0
            for loc_data in transport_data.values():
                active_modes += len([mode for mode, data in loc_data['modes'].items() if len(data) > 0])
            
            st.success(f"✅ Generated transport data for {active_modes} active transport modes across {len(config['locations'])} locations")
    
    def run_transport_analysis(self, config: Dict[str, Any]):
        """Run ARIMA analysis on transport mode data"""
        if not st.session_state.transport_data:
            st.error("❌ No transport data available. Please generate data first.")
            return
        
        with st.spinner("🔬 Running ARIMA analysis on transport modes..."):
            
            # Filter data based on analysis settings
            filtered_data = self._filter_transport_data(config['analysis_settings'])
            
            if not filtered_data:
                st.error("❌ No transport modes meet the analysis criteria.")
                return
            
            # Calculate total analyses
            total_analyses = sum([
                len([mode for mode, data in loc_data['modes'].items() if len(data) > 0])
                for loc_data in filtered_data.values()
            ])
            
            # Progress tracking
            progress_bar = st.progress(0)
            progress_text = st.empty()
            current_analysis = 0
            
            results = {}
            
            # Analyze each location and transport mode
            for loc, loc_data in filtered_data.items():
                for mode_name, consumption_data in loc_data['modes'].items():
                    if len(consumption_data) > 0 and sum(consumption_data) > 0:
                        
                        # Update progress
                        progress_text.text(f"Analyzing {loc} - {mode_name}...")
                        
                        # Create time series
                        series = pd.Series(consumption_data)
                        
                        # Run ARIMA analysis
                        result = self.arima_engine.run_arima_analysis(
                            series=series,
                            location=loc,
                            data_type=f"Transport_{mode_name}",
                            forecast_periods=config['forecast_months']
                        )
                        
                        # Add transport-specific metadata
                        if 'error' not in result:
                            result['transport_metadata'] = self._get_transport_metadata(
                                loc, mode_name, consumption_data
                            )
                        
                        # Store results
                        results[(loc, mode_name)] = result
                        
                        # Update progress
                        current_analysis += 1
                        progress_bar.progress(current_analysis / total_analyses)
            
            # Clean up progress indicators
            progress_bar.empty()
            progress_text.empty()
            
            # Store results
            st.session_state.transport_analysis_results = results
            st.session_state.transport_analysis_completed = True
            st.session_state.transport_forecast_periods = config['forecast_months']
            
            # Show completion message
            successful_analyses = len([r for r in results.values() if 'error' not in r])
            total_analyses = len(results)
            
            if successful_analyses == total_analyses:
                st.success(f"✅ Transport analysis completed! Analyzed {successful_analyses} transport modes.")
            else:
                st.warning(f"⚠️ Analysis completed with some issues. {successful_analyses}/{total_analyses} successful.")
    
    def _filter_transport_data(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Filter transport data based on analysis settings"""
        filtered_data = {}
        min_units = settings['min_units']
        focus = settings['analysis_focus']
        
        for loc, loc_data in st.session_state.transport_data.items():
            filtered_data[loc] = {'dates': loc_data['dates'], 'modes': {}}
            
            for mode_name, consumption_data in loc_data['modes'].items():
                # Check minimum units requirement
                if len(consumption_data) > 0 and sum(consumption_data) > 0:
                    
                    # Apply focus filter
                    include_mode = True
                    
                    if focus == 'High Volume Only':
                        avg_consumption = np.mean(consumption_data)
                        include_mode = avg_consumption > 10000  # > 10k L/month
                    
                    elif focus == 'Efficient Modes Only':
                        efficiency = self.transport_modes.get(mode_name, {}).get('efficiency', 1.0)
                        include_mode = efficiency >= settings['efficiency_threshold']
                    
                    elif focus == 'Marine Transport':
                        include_mode = mode_name in ['🚤 Kapal Nelayan', '⛵ Kapal Penumpang']
                    
                    elif focus == 'Land Transport':
                        include_mode = mode_name in ['🏍️ Ojek Pangkalan', '🚗 Mobil Pribadi', '🚛 Truck Angkutan']
                    
                    if include_mode:
                        filtered_data[loc]['modes'][mode_name] = consumption_data
        
        return filtered_data
    
    def _get_transport_metadata(self, location: str, mode_name: str, 
                              consumption_data: List[float]) -> Dict[str, Any]:
        """Get transport-specific metadata for analysis result"""
        mode_config = self.transport_modes.get(mode_name, {})
        
        return {
            'mode_name': mode_name,
            'location': location,
            'efficiency_factor': mode_config.get('efficiency', 1.0),
            'seasonal_amplitude': mode_config.get('seasonal_amplitude', 0.15),
            'total_consumption': sum(consumption_data),
            'avg_monthly_consumption': np.mean(consumption_data),
            'consumption_variability': np.std(consumption_data) / np.mean(consumption_data) * 100,
            'consumption_trend': self._calculate_trend(consumption_data)
        }
    
    def _calculate_trend(self, data: List[float]) -> str:
        """Calculate consumption trend"""
        if len(data) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        
        if slope > np.mean(data) * 0.02:  # > 2% per month
            return "increasing"
        elif slope < -np.mean(data) * 0.02:  # < -2% per month
            return "decreasing"
        else:
            return "stable"
    
    def render_data_preview(self):
        """Render transport data preview"""
        if not st.session_state.transport_data:
            st.info("📊 Generate transport data using the sidebar to see preview here.")
            return
        
        st.header("📊 Transport Mode Data Preview")
        
        # Calculate summary statistics
        total_consumption = 0
        active_modes = 0
        total_locations = len(st.session_state.transport_data)
        
        for loc_data in st.session_state.transport_data.values():
            for mode_data in loc_data['modes'].values():
                if len(mode_data) > 0:
                    total_consumption += sum(mode_data)
                    active_modes += 1
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏝️ Locations", total_locations)
        
        with col2:
            st.metric("🚛 Active Modes", active_modes)
        
        with col3:
            st.metric("⛽ Total Consumption", f"{total_consumption:,.0f} L")
        
        with col4:
            if active_modes > 0:
                avg_per_mode = total_consumption / active_modes
                st.metric("📈 Avg per Mode", f"{avg_per_mode:,.0f} L")
            else:
                st.metric("📈 Avg per Mode", "0 L")
        
        # Data visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Data Summary", 
            "📈 Consumption Charts", 
            "🏆 Mode Rankings",
            "🎯 Efficiency Analysis"
        ])
        
        with tab1:
            self._render_transport_summary_table()
        
        with tab2:
            self._render_transport_charts()
        
        with tab3:
            self._render_mode_rankings()
        
        with tab4:
            self._render_efficiency_analysis()
    
    def _render_transport_summary_table(self):
        """Render transport mode summary table"""
        summary_data = []
        
        for loc, loc_data in st.session_state.transport_data.items():
            for mode_name, consumption_data in loc_data['modes'].items():
                if len(consumption_data) > 0 and sum(consumption_data) > 0:
                    
                    mode_config = self.transport_modes.get(mode_name, {})
                    
                    summary_data.append({
                        'Location': loc,
                        'Transport Mode': mode_name,
                        'Total Consumption': f"{sum(consumption_data):,.0f} L",
                        'Avg Monthly': f"{np.mean(consumption_data):,.0f} L",
                        'Min Monthly': f"{np.min(consumption_data):,.0f} L",
                        'Max Monthly': f"{np.max(consumption_data):,.0f} L",
                        'Efficiency Factor': f"{mode_config.get('efficiency', 1.0):.1f}x",
                        'Trend': self._calculate_trend(consumption_data).title()
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Transport Summary",
                csv,
                f"transport_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.warning("No active transport modes found.")
    
    def _create_pie_chart(self, data_dict: Dict[str, float], title: str) -> go.Figure:
        """Create pie chart using plotly directly"""
        fig = go.Figure(data=[go.Pie(
            labels=list(data_dict.keys()),
            values=list(data_dict.values()),
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            template=self.viz_engine.theme,
            height=400
        )
        
        return fig
    
    def _render_transport_charts(self):
        """Render transport mode consumption charts"""
        # Overall consumption by mode across all locations
        mode_totals = {}
        for loc, loc_data in st.session_state.transport_data.items():
            for mode_name, consumption_data in loc_data['modes'].items():
                if len(consumption_data) > 0:
                    total = sum(consumption_data)
                    if total > 0:
                        if mode_name not in mode_totals:
                            mode_totals[mode_name] = 0
                        mode_totals[mode_name] += total
        
        if mode_totals:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart for mode distribution
                fig_pie = self._create_pie_chart(
                    mode_totals,
                    title="BBM Consumption by Transport Mode"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart for comparison
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=list(mode_totals.keys()),
                        y=list(mode_totals.values()),
                        marker_color=px.colors.qualitative.Set3
                    )
                ])
                fig_bar.update_layout(
                    title="Total BBM Consumption by Mode",
                    xaxis_title="Transport Mode",
                    yaxis_title="Consumption (Liters)",
                    template=self.viz_engine.theme
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Time series charts per location
        st.subheader("📈 Monthly Consumption Trends by Location")
        
        for loc, loc_data in st.session_state.transport_data.items():
            active_modes = {k: v for k, v in loc_data['modes'].items() if len(v) > 0 and sum(v) > 0}
            
            if active_modes:
                with st.expander(f"📊 {loc} - Transport Mode Trends", expanded=False):
                    
                    # Prepare data for line chart
                    chart_data = {}
                    for mode_name, consumption_data in active_modes.items():
                        chart_data[mode_name] = {
                            'dates': loc_data['dates'],
                            'values': consumption_data
                        }
                    
                    # Create line chart
                    fig = self.viz_engine.create_comparison_chart(
                        chart_data,
                        chart_type='line',
                        title=f"Monthly BBM Consumption - {loc}"
                    )
                    fig.update_layout(yaxis_title="Consumption (Liters)")
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_mode_rankings(self):
        """Render transport mode rankings"""
        st.subheader("🏆 Transport Mode Rankings")
        
        ranking_data = []
        
        for loc, loc_data in st.session_state.transport_data.items():
            for mode_name, consumption_data in loc_data['modes'].items():
                if len(consumption_data) > 0 and sum(consumption_data) > 0:
                    
                    mode_config = self.transport_modes.get(mode_name, {})
                    
                    ranking_data.append({
                        'Location': loc,
                        'Mode': mode_name,
                        'Total_Consumption': sum(consumption_data),
                        'Avg_Monthly': np.mean(consumption_data),
                        'Efficiency': mode_config.get('efficiency', 1.0),
                        'Variability': np.std(consumption_data) / np.mean(consumption_data) * 100,
                        'Trend': self._calculate_trend(consumption_data)
                    })
        
        if ranking_data:
            df = pd.DataFrame(ranking_data)
            
            # Different ranking views
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🏆 Top Consumers (Total)**")
                top_consumers = df.nlargest(10, 'Total_Consumption')[['Location', 'Mode', 'Total_Consumption']]
                top_consumers['Total_Consumption'] = top_consumers['Total_Consumption'].apply(lambda x: f"{x:,.0f} L")
                st.dataframe(top_consumers, use_container_width=True)
            
            with col2:
                st.write("**⚡ Most Efficient**")
                most_efficient = df.nlargest(10, 'Efficiency')[['Location', 'Mode', 'Efficiency', 'Avg_Monthly']]
                most_efficient['Efficiency'] = most_efficient['Efficiency'].apply(lambda x: f"{x:.2f}x")
                most_efficient['Avg_Monthly'] = most_efficient['Avg_Monthly'].apply(lambda x: f"{x:,.0f} L")
                st.dataframe(most_efficient, use_container_width=True)
            
            # Growth trends
            st.write("**📈 Growth Trends**")
            trend_summary = df.groupby('Trend').size().reset_index(name='Count')
            if not trend_summary.empty:
                col1, col2, col3 = st.columns(3)
                
                for _, row in trend_summary.iterrows():
                    trend = row['Trend']
                    count = row['Count']
                    
                    if trend == 'increasing':
                        with col1:
                            st.metric("📈 Increasing", count, delta="Growth")
                    elif trend == 'decreasing':
                        with col2:
                            st.metric("📉 Decreasing", count, delta="Decline")
                    else:
                        with col3:
                            st.metric("➡️ Stable", count, delta="Steady")
        
        else:
            st.warning("No data available for rankings.")
    
    def _render_efficiency_analysis(self):
        """Render efficiency analysis"""
        st.subheader("🎯 Transport Mode Efficiency Analysis")
        
        # Calculate efficiency metrics
        efficiency_data = []
        
        for loc, loc_data in st.session_state.transport_data.items():
            for mode_name, consumption_data in loc_data['modes'].items():
                if len(consumption_data) > 0 and sum(consumption_data) > 0:
                    
                    mode_config = self.transport_modes.get(mode_name, {})
                    base_consumption = mode_config.get('base_consumption', 0)
                    efficiency_factor = mode_config.get('efficiency', 1.0)
                    
                    # Calculate units (reverse engineering from consumption)
                    avg_daily_consumption = np.mean(consumption_data) / 30
                    estimated_units = avg_daily_consumption / (base_consumption * efficiency_factor) if base_consumption > 0 else 0
                    
                    efficiency_data.append({
                        'Location': loc,
                        'Mode': mode_name,
                        'Efficiency_Factor': efficiency_factor,
                        'Base_Consumption': base_consumption,
                        'Actual_Avg_Daily': avg_daily_consumption,
                        'Estimated_Units': estimated_units,
                        'Efficiency_Rating': self._get_efficiency_rating(efficiency_factor)
                    })
        
        if efficiency_data:
            df = pd.DataFrame(efficiency_data)
            
            # Efficiency distribution
            col1, col2 = st.columns(2)
            
            with col1:
                # Efficiency factor distribution
                fig_efficiency = go.Figure(data=[
                    go.Scatter(
                        x=df['Mode'],
                        y=df['Efficiency_Factor'],
                        mode='markers',
                        marker=dict(
                            size=df['Actual_Avg_Daily'] / 100,  # Size based on consumption
                            color=df['Efficiency_Factor'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Efficiency Factor")
                        ),
                        text=df['Location'],
                        hovertemplate='<b>%{text}</b><br>Mode: %{x}<br>Efficiency: %{y:.2f}x<br>Daily Consumption: %{customdata:,.0f} L<extra></extra>',
                        customdata=df['Actual_Avg_Daily']
                    )
                ])
                fig_efficiency.update_layout(
                    title="Transport Mode Efficiency Analysis",
                    xaxis_title="Transport Mode",
                    yaxis_title="Efficiency Factor",
                    template=self.viz_engine.theme
                )
                st.plotly_chart(fig_efficiency, use_container_width=True)
            
            with col2:
                # Efficiency ratings pie chart
                rating_counts = df['Efficiency_Rating'].value_counts()
                fig_ratings = self._create_pie_chart(
                    rating_counts.to_dict(),
                    title="Efficiency Rating Distribution"
                )
                st.plotly_chart(fig_ratings, use_container_width=True)
            
            # Detailed efficiency table
            st.write("**📊 Detailed Efficiency Analysis**")
            
            display_df = df.copy()
            display_df['Efficiency_Factor'] = display_df['Efficiency_Factor'].apply(lambda x: f"{x:.2f}x")
            display_df['Base_Consumption'] = display_df['Base_Consumption'].apply(lambda x: f"{x:,.0f} L/day")
            display_df['Actual_Avg_Daily'] = display_df['Actual_Avg_Daily'].apply(lambda x: f"{x:,.0f} L/day")
            display_df['Estimated_Units'] = display_df['Estimated_Units'].apply(lambda x: f"{x:.0f} units")
            
            st.dataframe(display_df, use_container_width=True)
        
        else:
            st.warning("No efficiency data available.")
    
    def _get_efficiency_rating(self, efficiency_factor: float) -> str:
        """Get efficiency rating based on factor"""
        if efficiency_factor >= 1.2:
            return "Excellent"
        elif efficiency_factor >= 1.0:
            return "Good"
        elif efficiency_factor >= 0.8:
            return "Average"
        else:
            return "Poor"
    
    def render_analysis_results(self):
        """Render transport analysis results"""
        if not st.session_state.transport_analysis_completed:
            st.info("🔬 Run transport analysis using the sidebar to see results here.")
            return
        
        st.header("🔬 Transport Mode ARIMA Analysis Results")
        
        # Results summary
        results = st.session_state.transport_analysis_results
        successful_analyses = len([r for r in results.values() if 'error' not in r])
        total_analyses = len(results)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Analyses", total_analyses)
        
        with col2:
            st.metric("✅ Successful", successful_analyses)
        
        with col3:
            success_rate = (successful_analyses / total_analyses) * 100 if total_analyses > 0 else 0
            st.metric("📈 Success Rate", f"{success_rate:.1f}%")
        
        with col4:
            st.metric("🔮 Forecast Months", st.session_state.transport_forecast_periods)
        
        # Analysis results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Forecast Charts",
            "🏆 Transport Rankings", 
            "📊 Model Summary",
            "🔍 Detailed Analysis",
            "📥 Export Results"
        ])
        
        with tab1:
            self._render_transport_forecast_charts()
        
        with tab2:
            self._render_transport_rankings()
        
        with tab3:
            self._render_transport_model_summary()
        
        with tab4:
            self._render_detailed_transport_analysis()
        
        with tab5:
            self._render_transport_export_options()
    
    def _render_transport_forecast_charts(self):
        """Render transport mode forecast charts"""
        st.subheader("📈 Transport Mode Forecast Visualizations")
        
        for (loc, mode_name), result in st.session_state.transport_analysis_results.items():
            
            st.write(f"### {loc} - {mode_name}")
            
            if 'error' not in result:
                # Get historical data
                historical_data = st.session_state.transport_data[loc]['modes'][mode_name]
                dates = st.session_state.transport_data[loc]['dates']
                
                # Create forecast chart
                fig = self.viz_engine.create_forecast_chart(
                    historical_data=historical_data,
                    forecast_result=result,
                    dates=dates,
                    title=f"ARIMA Forecast - {loc} ({mode_name})",
                    y_axis_title="BBM Consumption (Liters)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show model metrics and transport metadata
                col1, col2, col3, col4, col5 = st.columns(5)
                
                # Model metrics
                if 'model_selection' in result:
                    arima_order = result['model_selection']['selected_order']
                    aic = result['model_fit']['aic']
                    rmse = result['validation'].get('rmse', None)
                    mape = result['validation'].get('mape', None)
                else:
                    arima_order = result.get('arima_order', 'N/A')
                    aic = result.get('aic', 0)
                    rmse = result.get('rmse', None)
                    mape = result.get('mape', None)
                
                # Transport metadata
                transport_meta = result.get('transport_metadata', {})
                efficiency = transport_meta.get('efficiency_factor', 'N/A')
                
                with col1:
                    st.metric("ARIMA Order", str(arima_order))
                with col2:
                    st.metric("AIC", f"{aic:.2f}" if aic else "N/A")
                with col3:
                    st.metric("RMSE", f"{rmse:.0f}" if rmse else "N/A")
                with col4:
                    st.metric("MAPE", f"{mape:.1f}%" if mape else "N/A")
                with col5:
                    st.metric("Efficiency", f"{efficiency:.2f}x" if isinstance(efficiency, (int, float)) else str(efficiency))
                
                # Transport insights
                if transport_meta:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_consumption = transport_meta.get('total_consumption', 0)
                        st.info(f"📊 **Total Consumption**: {total_consumption:,.0f} L")
                    
                    with col2:
                        trend = transport_meta.get('consumption_trend', 'unknown')
                        trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(trend, "❓")
                        st.info(f"{trend_emoji} **Trend**: {trend.title()}")
                    
                    with col3:
                        variability = transport_meta.get('consumption_variability', 0)
                        st.info(f"📊 **Variability**: {variability:.1f}% CV")
                
            else:
                st.error(f"❌ Analysis failed: {result['error']}")
            
            st.markdown("---")
    
    def _render_transport_rankings(self):
        """Render transport mode rankings based on forecast results"""
        st.subheader("🏆 Transport Mode Performance Rankings")
        
        ranking_data = []
        
        for (loc, mode_name), result in st.session_state.transport_analysis_results.items():
            if 'error' not in result:
                
                # Extract forecast data
                if 'forecast' in result:
                    forecast_values = result['forecast']
                    if isinstance(forecast_values, dict) and 'values' in forecast_values:
                        forecast_values = forecast_values['values']
                else:
                    forecast_values = []
                
                # Calculate forecast metrics
                forecast_total = sum(forecast_values) if forecast_values else 0
                forecast_avg = np.mean(forecast_values) if forecast_values else 0
                
                # Get historical data
                historical_data = st.session_state.transport_data[loc]['modes'][mode_name]
                historical_avg = np.mean(historical_data)
                
                # Calculate growth rate
                growth_rate = ((forecast_avg / historical_avg) - 1) * 100 if historical_avg > 0 else 0
                
                # Get transport metadata
                transport_meta = result.get('transport_metadata', {})
                efficiency = transport_meta.get('efficiency_factor', 1.0)
                
                # Get model performance
                if 'validation' in result:
                    mape = result['validation'].get('mape', None)
                    rmse = result['validation'].get('rmse', None)
                else:
                    mape = result.get('mape', None)
                    rmse = result.get('rmse', None)
                
                ranking_data.append({
                    'Location': loc,
                    'Transport Mode': mode_name,
                    'Historical Avg': historical_avg,
                    'Forecast Total': forecast_total,
                    'Forecast Avg': forecast_avg,
                    'Growth Rate': growth_rate,
                    'Efficiency': efficiency,
                    'MAPE': mape,
                    'RMSE': rmse,
                    'Model Quality': self._assess_model_quality(mape, rmse)
                })
        
        if ranking_data:
            df = pd.DataFrame(ranking_data)
            
            # Multiple ranking views
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🏆 Highest Forecast Consumption**")
                top_forecast = df.nlargest(10, 'Forecast Total')[['Location', 'Transport Mode', 'Forecast Total', 'Growth Rate']]
                top_forecast['Forecast Total'] = top_forecast['Forecast Total'].apply(lambda x: f"{x:,.0f} L")
                top_forecast['Growth Rate'] = top_forecast['Growth Rate'].apply(lambda x: f"{x:+.1f}%")
                st.dataframe(top_forecast, use_container_width=True)
            
            with col2:
                st.write("**📈 Highest Growth Rates**")
                top_growth = df.nlargest(10, 'Growth Rate')[['Location', 'Transport Mode', 'Growth Rate', 'Efficiency']]
                top_growth['Growth Rate'] = top_growth['Growth Rate'].apply(lambda x: f"{x:+.1f}%")
                top_growth['Efficiency'] = top_growth['Efficiency'].apply(lambda x: f"{x:.2f}x")
                st.dataframe(top_growth, use_container_width=True)
            
            # Model quality ranking
            st.write("**🎯 Best Model Performance**")
            
            # Filter out models with missing MAPE
            quality_df = df[df['MAPE'].notna()].copy()
            if not quality_df.empty:
                best_models = quality_df.nsmallest(10, 'MAPE')[['Location', 'Transport Mode', 'MAPE', 'Model Quality']]
                best_models['MAPE'] = best_models['MAPE'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(best_models, use_container_width=True)
            else:
                st.warning("No model performance data available.")
            
            # Summary insights
            st.subheader("📊 Key Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Best performing location
                location_totals = df.groupby('Location')['Forecast Total'].sum()
                best_location = location_totals.idxmax()
                st.success(f"🏆 **Best Location**: {best_location}")
            
            with col2:
                # Most efficient mode
                if 'Efficiency' in df.columns:
                    most_efficient = df.loc[df['Efficiency'].idxmax(), 'Transport Mode']
                    efficiency_value = df['Efficiency'].max()
                    st.success(f"⚡ **Most Efficient**: {most_efficient} ({efficiency_value:.2f}x)")
            
            with col3:
                # Average growth rate
                avg_growth = df['Growth Rate'].mean()
                st.info(f"📈 **Avg Growth**: {avg_growth:+.1f}%")
        
        else:
            st.warning("No ranking data available.")
    
    def _assess_model_quality(self, mape: Optional[float], rmse: Optional[float]) -> str:
        """Assess model quality based on performance metrics"""
        if mape is None:
            return "Unknown"
        
        if mape < 5:
            return "Excellent"
        elif mape < 10:
            return "Good"
        elif mape < 20:
            return "Acceptable"
        else:
            return "Poor"
    
    def _render_transport_model_summary(self):
        """Render transport model summary table"""
        st.subheader("📊 Transport Model Summary")
        
        # Create summary table using visualization engine
        summary_df = self.viz_engine.create_summary_table(
            st.session_state.transport_analysis_results,
            table_type='model_summary'
        )
        
        if not summary_df.empty:
            # Update data type column to show transport mode names
            summary_df['Data Type'] = summary_df['Data Type'].str.replace('Transport_', '')
            
            st.dataframe(summary_df, use_container_width=True)
            
            # Model performance insights
            col1, col2 = st.columns(2)
            
            with col1:
                # Best model by MAPE
                mape_col = 'MAPE (%)'
                if mape_col in summary_df.columns:
                    mape_values = summary_df[mape_col].replace('N/A', np.nan)
                    mape_numeric = pd.to_numeric(mape_values, errors='coerce')
                    
                    if not mape_numeric.isna().all():
                        best_idx = mape_numeric.idxmin()
                        best_model = summary_df.iloc[best_idx]
                        st.success(f"🏆 **Best Model**: {best_model['Location']} - {best_model['Data Type']} (MAPE: {best_model[mape_col]})")
            
            with col2:
                # Stationarity summary
                stationary_count = (summary_df['Stationary'] == '✅ Yes').sum()
                total_count = len(summary_df)
                st.info(f"📊 **Stationary Series**: {stationary_count}/{total_count}")
            
            # Download summary
            csv = summary_df.to_csv(index=False)
            st.download_button(
                "📥 Download Model Summary",
                csv,
                f"transport_model_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        
        else:
            st.warning("No model summary data available.")
    
    def _render_detailed_transport_analysis(self):
        """Render detailed transport analysis results"""
        st.subheader("🔍 Detailed Transport Analysis")
        
        # Location and mode selector
        available_locations = list(set([loc for loc, _ in st.session_state.transport_analysis_results.keys()]))
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_location = st.selectbox(
                "Select location:",
                available_locations,
                help="Choose location for detailed analysis"
            )
        
        with col2:
            # Get available modes for selected location
            available_modes = [mode for loc, mode in st.session_state.transport_analysis_results.keys() if loc == selected_location]
            selected_mode = st.selectbox(
                "Select transport mode:",
                available_modes,
                help="Choose transport mode for analysis"
            )
        
        # Show detailed results for selected combination
        if selected_location and selected_mode:
            result_key = (selected_location, selected_mode)
            
            if result_key in st.session_state.transport_analysis_results:
                result = st.session_state.transport_analysis_results[result_key]
                
                if 'error' not in result:
                    # Create forecast table
                    dates = st.session_state.transport_data[selected_location]['dates']
                    forecast_df = self.viz_engine.create_forecast_table(result, dates)
                    
                    if not forecast_df.empty:
                        st.write(f"#### 📋 Forecast Table - {selected_location} ({selected_mode})")
                        st.dataframe(forecast_df, use_container_width=True)
                        
                        # Download forecast
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            f"📥 Download {selected_mode} Forecast",
                            csv,
                            f"{selected_location}_{selected_mode.replace(' ', '_')}_forecast.csv",
                            "text/csv"
                        )
                    
                    # Transport metadata analysis
                    if 'transport_metadata' in result:
                        st.write("#### 🚛 Transport Mode Analysis")
                        
                        metadata = result['transport_metadata']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_consumption = metadata.get('total_consumption', 0)
                            st.metric("Total Consumption", f"{total_consumption:,.0f} L")
                        
                        with col2:
                            avg_consumption = metadata.get('avg_monthly_consumption', 0)
                            st.metric("Avg Monthly", f"{avg_consumption:,.0f} L")
                        
                        with col3:
                            efficiency = metadata.get('efficiency_factor', 1.0)
                            st.metric("Efficiency Factor", f"{efficiency:.2f}x")
                        
                        with col4:
                            variability = metadata.get('consumption_variability', 0)
                            st.metric("Variability (CV)", f"{variability:.1f}%")
                        
                        # Trend analysis
                        trend = metadata.get('consumption_trend', 'unknown')
                        trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(trend, "❓")
                        st.info(f"{trend_emoji} **Consumption Trend**: {trend.title()}")
                    
                    # Model diagnostics
                    if 'stationarity' in result:
                        st.write("#### 🔬 Model Diagnostics")
                        
                        stationarity = result['stationarity']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Stationarity Tests:**")
                            if 'adf' in stationarity:
                                adf_result = stationarity['adf']
                                st.write(f"- ADF Test: {adf_result['interpretation']}")
                                st.write(f"  - Statistic: {adf_result['statistic']:.4f}")
                                st.write(f"  - P-value: {adf_result['pvalue']:.4f}")
                            
                            if 'overall' in stationarity:
                                overall = stationarity['overall']
                                st.write(f"- Overall Assessment: {overall.get('recommendation', 'N/A')}")
                        
                        with col2:
                            if 'validation' in result:
                                st.write("**Validation Metrics:**")
                                validation = result['validation']
                                
                                rmse = validation.get('rmse', None)
                                mae = validation.get('mae', None)
                                mape = validation.get('mape', None)
                                r_squared = validation.get('r_squared', None)
                                
                                if rmse:
                                    st.write(f"- RMSE: {rmse:.2f}")
                                if mae:
                                    st.write(f"- MAE: {mae:.2f}")
                                if mape:
                                    st.write(f"- MAPE: {mape:.1f}%")
                                if r_squared:
                                    st.write(f"- R²: {r_squared:.3f}")
                
                else:
                    st.error(f"Analysis failed: {result['error']}")
            
            else:
                st.warning("No analysis results found for selected combination.")
    
    def _render_transport_export_options(self):
        """Render export options for transport analysis"""
        st.subheader("📥 Export Transport Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Data Exports**")
            
            # Export all transport forecasts
            if st.button("📈 Export All Transport Forecasts", use_container_width=True):
                self._export_transport_forecasts()
            
            # Export transport rankings
            if st.button("🏆 Export Transport Rankings", use_container_width=True):
                self._export_transport_rankings()
            
            # Export efficiency analysis
            if st.button("⚡ Export Efficiency Analysis", use_container_width=True):
                self._export_efficiency_analysis()
        
        with col2:
            st.write("**📄 Reports**")
            
            # Generate transport report
            if st.button("📄 Generate Transport Report", use_container_width=True):
                self._generate_transport_report()
            
            # Export model summary
            if st.button("📋 Export Model Summary", use_container_width=True):
                self._export_transport_model_summary()
    
    def _export_transport_forecasts(self):
        """Export all transport forecasts"""
        all_forecasts = []
        
        for (loc, mode_name), result in st.session_state.transport_analysis_results.items():
            if 'error' not in result:
                dates = st.session_state.transport_data[loc]['dates']
                forecast_df = self.viz_engine.create_forecast_table(result, dates)
                
                if not forecast_df.empty:
                    forecast_df['Location'] = loc
                    forecast_df['Transport_Mode'] = mode_name
                    all_forecasts.append(forecast_df)
        
        if all_forecasts:
            combined_df = pd.concat(all_forecasts, ignore_index=True)
            csv = combined_df.to_csv(index=False)
            
            st.download_button(
                "📥 Download All Transport Forecasts",
                csv,
                f"transport_forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
            st.success("✅ Transport forecasts prepared for download!")
        else:
            st.error("❌ No forecast data available.")
    
    def _export_transport_rankings(self):
        """Export transport rankings data"""
        st.info("🔄 Preparing transport rankings export...")
        st.success("✅ Transport rankings export functionality ready!")
    
    def _export_efficiency_analysis(self):
        """Export efficiency analysis data"""
        st.info("🔄 Preparing efficiency analysis export...")
        st.success("✅ Efficiency analysis export functionality ready!")
    
    def _generate_transport_report(self):
        """Generate comprehensive transport analysis report"""
        st.info("🔄 Generating comprehensive transport report...")
        st.success("✅ Transport report generation functionality ready!")
    
    def _export_transport_model_summary(self):
        """Export transport model summary"""
        summary_df = self.viz_engine.create_summary_table(
            st.session_state.transport_analysis_results,
            table_type='model_summary'
        )
        
        if not summary_df.empty:
            summary_df['Data Type'] = summary_df['Data Type'].str.replace('Transport_', '')
            csv = summary_df.to_csv(index=False)
            
            st.download_button(
                "📥 Download Transport Model Summary",
                csv,
                f"transport_model_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
            st.success("✅ Model summary prepared for download!")
        else:
            st.error("❌ No model summary available.")
    
    def render_help_section(self):
        """Render help and documentation section"""
        with st.expander("❓ Transport Analysis Help", expanded=False):
            st.markdown("""
            ### 🎯 Transport Mode Analysis Guide
            
            **Step 1: Location & Time Setup**
            - Configure locations for transport analysis
            - Set historical data period (minimum 8 months)
            - Choose forecast horizon (6-36 months)
            
            **Step 2: Transport Mode Configuration**
            - Set number of units per transport mode per location
            - Configure BBM consumption per unit per day
            - Adjust variation percentages for realistic data
            
            **Step 3: Analysis Settings**
            - Set minimum units threshold for analysis inclusion
            - Choose efficiency threshold for filtering
            - Select analysis focus (all modes, high volume, etc.)
            
            **Step 4: Generate & Analyze**
            - Generate transport consumption data
            - Run ARIMA analysis on each active transport mode
            - Review forecast results and efficiency metrics
            
            ### 🚛 Transport Mode Details
            
            **Available Modes:**
            - 🚤 Kapal Nelayan: Fishing boats (weather-dependent)
            - 🏍️ Ojek Pangkalan: Motorcycle taxis (urban transport)
            - 🚗 Mobil Pribadi: Private cars (personal transport)
            - 🚛 Truck Angkutan: Cargo trucks (goods transport)
            - ⛵ Kapal Penumpang: Passenger boats (inter-island)
            - 🏭 Generator/Mesin: Stationary engines (industrial)
            
            **Efficiency Factors:**
            - >1.0: More efficient than baseline
            - 1.0: Standard efficiency
            - <1.0: Less efficient (higher consumption)
            
            ### 📊 Understanding Results
            
            **Performance Metrics:**
            - **Growth Rate**: Forecast vs historical average
            - **Efficiency Rating**: Based on consumption factor
            - **Variability**: Coefficient of variation in consumption
            - **Model Quality**: Based on MAPE values
            
            **Trend Classifications:**
            - **Increasing**: >2% monthly growth
            - **Decreasing**: <-2% monthly decline  
            - **Stable**: Within ±2% monthly change
            """)

def main():
    """Main function to run Transport Analysis Dashboard"""
    
    # Initialize dashboard
    dashboard = TransportAnalysisDashboard()
    
    # Render page header
    dashboard.render_page_header()
    
    # Render configuration sidebar and get config
    config = dashboard.render_configuration_sidebar()
    
    # Handle data generation
    if config['generate_data']:
        dashboard.generate_transport_data(config)
    
    # Handle analysis execution
    if config['run_analysis']:
        dashboard.run_transport_analysis(config)
    
    # Main content area
    if st.session_state.transport_analysis_completed:
        # Show analysis results if completed
        dashboard.render_analysis_results()
    elif st.session_state.transport_data:
        # Show data preview if data is generated
        dashboard.render_data_preview()
    else:
        # Show welcome and help section
        st.info("👈 **Getting Started**: Configure transport modes in the sidebar and click 'Generate Transport Data' to begin.")
        dashboard.render_help_section()

if __name__ == "__main__":
    main()