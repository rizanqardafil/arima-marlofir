"""
BBM Analysis Dashboard - Main Analysis Page
/Users/sociolla/Documents/BBM/pages/bbm_analysis.py

Comprehensive BBM analysis dashboard integrating all core engines
Provides complete BBM forecasting workflow with ARIMA analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.arima_engine import ARIMAEngine
from core.data_generator import DataGenerator, generate_date_range
from core.visualization import VisualizationEngine

class BBMAnalysisDashboard:
    """
    Complete BBM Analysis Dashboard with integrated workflow
    
    Features:
    - Multi-location BBM data configuration
    - Real-time data generation with seasonal patterns
    - ARIMA analysis pipeline
    - Interactive visualizations
    - Export capabilities
    """
    
    def __init__(self):
        """Initialize BBM Analysis Dashboard"""
        self.initialize_session_state()
        self.arima_engine = ARIMAEngine(max_p=3, max_d=2, max_q=3)
        self.data_generator = DataGenerator(random_seed=42)
        self.viz_engine = VisualizationEngine(theme='plotly_white')
        
        # Default configuration
        self.default_locations = ["Jemaja", "Siantan", "Palmatak", "Kute Siantan", "Siantan Timur"]
        self.bbm_types = ['BBM Tipe 1', 'BBM Tipe 2']
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        session_vars = {
            'bbm_config': {},
            'bbm_data': {},
            'analysis_results': {},
            'analysis_completed': False,
            'forecast_periods': 12
        }
        
        for var, default_value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default_value
    
    def render_page_header(self):
        """Render page header with title and description"""
        st.title("⛽ BBM Analysis Dashboard")
        st.markdown("""
        **Comprehensive BBM Consumption Forecasting with ARIMA Analysis**
        
        Analyze BBM consumption patterns across multiple locations with advanced time series forecasting.
        Generate realistic data, run ARIMA analysis, and create interactive visualizations.
        """)
        st.markdown("---")
    
    def render_configuration_sidebar(self) -> Dict[str, Any]:
        """
        Render configuration sidebar for BBM analysis
        
        Returns:
            Configuration dictionary
        """
        with st.sidebar:
            st.header("⚙️ BBM Analysis Configuration")
            
            # Location Configuration
            st.subheader("1. 📍 Location Setup")
            num_locations = st.number_input(
                "Number of locations to analyze:",
                min_value=1, max_value=5, value=2,
                help="Choose how many locations to include in the analysis"
            )
            
            locations = []
            for i in range(num_locations):
                default_name = self.default_locations[i] if i < len(self.default_locations) else f"Location_{i+1}"
                loc_name = st.text_input(
                    f"Location {i+1} name:",
                    value=default_name,
                    key=f"bbm_location_{i}",
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
                    help="Number of months of historical data to generate (minimum 8 for ARIMA)"
                )
            
            with col2:
                forecast_months = st.slider(
                    "Forecast period (months):",
                    min_value=6, max_value=36, value=12,
                    help="Number of months to forecast into the future"
                )
            
            # Date range selection
            start_date = st.date_input(
                "Historical data start date:",
                value=datetime(2023, 1, 1),
                help="Starting date for historical data"
            )
            
            st.markdown("---")
            
            # BBM Data Configuration
            st.subheader("3. ⛽ BBM Data Configuration")
            
            bbm_params = self._render_bbm_parameters(locations)
            
            st.markdown("---")
            
            # Advanced Settings
            with st.expander("🔧 Advanced ARIMA Settings", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_p = st.number_input("Max AR order (p):", min_value=1, max_value=5, value=3)
                with col2:
                    max_d = st.number_input("Max differencing (d):", min_value=1, max_value=3, value=2)
                with col3:
                    max_q = st.number_input("Max MA order (q):", min_value=1, max_value=5, value=3)
                
                information_criterion = st.selectbox(
                    "Model selection criterion:",
                    options=['AIC', 'BIC', 'HQIC'],
                    index=0,
                    help="Information criterion for ARIMA model selection"
                )
            
            st.markdown("---")
            
            # Action Buttons
            st.subheader("4. 🚀 Analysis Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                generate_data_btn = st.button(
                    "📊 Generate Data",
                    help="Generate BBM data with current configuration",
                    use_container_width=True
                )
            
            with col2:
                run_analysis_btn = st.button(
                    "🔬 Run ARIMA Analysis",
                    help="Run ARIMA analysis on generated data",
                    use_container_width=True,
                    disabled=len(st.session_state.bbm_data) == 0
                )
            
            # Store configuration
            config = {
                'locations': locations,
                'num_months': num_months,
                'forecast_months': forecast_months,
                'start_date': datetime.combine(start_date, datetime.min.time()),
                'bbm_params': bbm_params,
                'arima_params': {
                    'max_p': max_p,
                    'max_d': max_d,
                    'max_q': max_q,
                    'criterion': information_criterion
                },
                'generate_data': generate_data_btn,
                'run_analysis': run_analysis_btn
            }
            
            return config
    
    def _render_bbm_parameters(self, locations: List[str]) -> Dict[str, Any]:
        """Render BBM parameter configuration for each location"""
        bbm_params = {}
        
        for loc in locations:
            with st.expander(f"📍 {loc} - BBM Configuration", expanded=False):
                
                # BBM Tipe 1 Configuration
                st.write("**BBM Tipe 1 (Premium/Ron 92)**")
                col1, col2 = st.columns(2)
                
                with col1:
                    bbm_params[f"{loc}_base_t1"] = st.number_input(
                        "Base consumption (L/month):",
                        min_value=1000, max_value=50000, value=5000, step=500,
                        key=f"{loc}_base_t1",
                        help="Average monthly consumption for BBM Tipe 1"
                    )
                
                with col2:
                    bbm_params[f"{loc}_var_t1"] = st.slider(
                        "Variation (±%):",
                        min_value=5, max_value=50, value=20,
                        key=f"{loc}_var_t1",
                        help="Percentage variation around base consumption"
                    )
                
                # BBM Tipe 2 Configuration
                st.write("**BBM Tipe 2 (Solar/Diesel)**")
                col1, col2 = st.columns(2)
                
                with col1:
                    bbm_params[f"{loc}_base_t2"] = st.number_input(
                        "Base consumption (L/month):",
                        min_value=2000, max_value=100000, value=10000, step=1000,
                        key=f"{loc}_base_t2",
                        help="Average monthly consumption for BBM Tipe 2"
                    )
                
                with col2:
                    bbm_params[f"{loc}_var_t2"] = st.slider(
                        "Variation (±%):",
                        min_value=5, max_value=50, value=25,
                        key=f"{loc}_var_t2",
                        help="Percentage variation around base consumption"
                    )
                
                # Seasonal Pattern Settings - FIXED: No nested expander
                st.write("**🌊 Seasonal Pattern Settings**")
                
                col1, col2 = st.columns(2)
                with col1:
                    bbm_params[f"{loc}_seasonal_t1"] = st.slider(
                        "BBM Tipe 1 seasonal amplitude:",
                        min_value=0.1, max_value=0.5, value=0.2, step=0.05,
                        key=f"{loc}_seasonal_t1",
                        help="Strength of seasonal variation (0.1 = 10% variation)"
                    )
                
                with col2:
                    bbm_params[f"{loc}_seasonal_t2"] = st.slider(
                        "BBM Tipe 2 seasonal amplitude:",
                        min_value=0.1, max_value=0.5, value=0.3, step=0.05,
                        key=f"{loc}_seasonal_t2",
                        help="Strength of seasonal variation (0.1 = 10% variation)"
                    )
                
                bbm_params[f"{loc}_peak_month"] = st.selectbox(
                    "Peak consumption month:",
                    options=list(range(1, 13)),
                    index=6,  # July
                    format_func=lambda x: datetime(2023, x, 1).strftime('%B'),
                    key=f"{loc}_peak_month",
                    help="Month when consumption typically peaks"
                )
        
        return bbm_params    
    
    def generate_bbm_data(self, config: Dict[str, Any]):
        """Generate BBM data based on configuration"""
        with st.spinner("🔄 Generating BBM data with seasonal patterns..."):
            
            # Generate date range
            dates = generate_date_range(config['start_date'], config['num_months'])
            
            # Generate BBM data
            bbm_data = self.data_generator.generate_bbm_data(
                locations=config['locations'],
                num_months=config['num_months'],
                parameters=config['bbm_params']
            )
            
            # Store data with dates
            for loc in config['locations']:
                st.session_state.bbm_data[loc] = {
                    'dates': dates,
                    'bbm_tipe_1': bbm_data[loc]['bbm_tipe_1'],
                    'bbm_tipe_2': bbm_data[loc]['bbm_tipe_2']
                }
            
            # Store configuration
            st.session_state.bbm_config = config
            
            st.success(f"✅ Generated BBM data for {len(config['locations'])} locations over {config['num_months']} months")
    
    def run_arima_analysis(self, config: Dict[str, Any]):
        """Run ARIMA analysis on generated BBM data"""
        if not st.session_state.bbm_data:
            st.error("❌ No BBM data available. Please generate data first.")
            return
        
        # Update ARIMA engine parameters
        self.arima_engine.max_p = config['arima_params']['max_p']
        self.arima_engine.max_d = config['arima_params']['max_d']
        self.arima_engine.max_q = config['arima_params']['max_q']
        
        with st.spinner("🔬 Running ARIMA analysis... This may take a few moments..."):
            
            # Calculate total number of analyses
            total_analyses = len(st.session_state.bbm_data) * len(self.bbm_types)
            
            # Progress tracking
            progress_bar = st.progress(0)
            progress_text = st.empty()
            current_analysis = 0
            
            results = {}
            
            # Analyze each location and BBM type
            for loc, loc_data in st.session_state.bbm_data.items():
                for bbm_type in self.bbm_types:
                    
                    # Update progress
                    progress_text.text(f"Analyzing {loc} - {bbm_type}...")
                    
                    # Get time series data
                    if bbm_type == 'BBM Tipe 1':
                        series = pd.Series(loc_data['bbm_tipe_1'])
                    else:
                        series = pd.Series(loc_data['bbm_tipe_2'])
                    
                    # Run ARIMA analysis
                    result = self.arima_engine.run_arima_analysis(
                        series=series,
                        location=loc,
                        data_type=bbm_type,
                        forecast_periods=config['forecast_months']
                    )
                    
                    # Store results
                    results[(loc, bbm_type)] = result
                    
                    # Update progress
                    current_analysis += 1
                    progress_bar.progress(current_analysis / total_analyses)
            
            # Clean up progress indicators
            progress_bar.empty()
            progress_text.empty()
            
            # Store results
            st.session_state.analysis_results = results
            st.session_state.analysis_completed = True
            st.session_state.forecast_periods = config['forecast_months']
            
            # Show completion message
            successful_analyses = len([r for r in results.values() if 'error' not in r])
            total_analyses = len(results)
            
            if successful_analyses == total_analyses:
                st.success(f"✅ ARIMA analysis completed successfully! Analyzed {successful_analyses} time series.")
            else:
                st.warning(f"⚠️ ARIMA analysis completed with some issues. {successful_analyses}/{total_analyses} analyses successful.")
    
    def render_data_preview(self):
        """Render data preview section"""
        if not st.session_state.bbm_data:
            st.info("📊 Generate BBM data using the sidebar configuration to see preview here.")
            return
        
        st.header("📊 Generated BBM Data Preview")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_locations = len(st.session_state.bbm_data)
        total_months = len(list(st.session_state.bbm_data.values())[0]['dates'])
        
        with col1:
            st.metric("📍 Locations", total_locations)
        
        with col2:
            st.metric("📅 Months", total_months)
        
        with col3:
            # Calculate total consumption
            total_consumption = sum([
                sum(data['bbm_tipe_1']) + sum(data['bbm_tipe_2'])
                for data in st.session_state.bbm_data.values()
            ])
            st.metric("⛽ Total Consumption", f"{total_consumption:,.0f} L")
        
        with col4:
            avg_monthly = total_consumption / (total_locations * total_months)
            st.metric("📈 Avg Monthly", f"{avg_monthly:,.0f} L")
        
        # Data table and visualization
        tab1, tab2, tab3 = st.tabs(["📋 Data Table", "📈 Time Series Charts", "📊 Summary Statistics"])
        
        with tab1:
            self._render_data_table()
        
        with tab2:
            self._render_time_series_charts()
        
        with tab3:
            self._render_summary_statistics()
    
    def _render_data_table(self):
        """Render BBM data table"""
        # Prepare data for table
        table_data = []
        
        for loc, data in st.session_state.bbm_data.items():
            for i, date in enumerate(data['dates']):
                table_data.append({
                    'Date': date.strftime('%b %Y'),
                    'Location': loc,
                    'BBM Tipe 1 (L)': f"{data['bbm_tipe_1'][i]:,.0f}",
                    'BBM Tipe 2 (L)': f"{data['bbm_tipe_2'][i]:,.0f}",
                    'Total (L)': f"{data['bbm_tipe_1'][i] + data['bbm_tipe_2'][i]:,.0f}"
                })
        
        df = pd.DataFrame(table_data)
        
        # Display with pagination
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data Table (CSV)",
            data=csv,
            file_name=f"bbm_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    def _render_time_series_charts(self):
        """Render time series visualization charts"""
        # Prepare data for visualization
        chart_data = {}
        
        for loc, data in st.session_state.bbm_data.items():
            for bbm_type in self.bbm_types:
                series_name = f"{loc} - {bbm_type}"
                if bbm_type == 'BBM Tipe 1':
                    values = data['bbm_tipe_1']
                else:
                    values = data['bbm_tipe_2']
                
                chart_data[series_name] = {
                    'dates': data['dates'],
                    'values': values
                }
        
        # Create comparison chart
        fig = self.viz_engine.create_comparison_chart(
            chart_data,
            chart_type='line',
            title="BBM Consumption Time Series - All Locations"
        )
        
        fig.update_layout(
            yaxis_title="Consumption (Liters)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual location charts
        st.subheader("📍 Individual Location Charts")
        
        for loc, data in st.session_state.bbm_data.items():
            with st.expander(f"📊 {loc} - Detailed View", expanded=False):
                
                loc_chart_data = {
                    'BBM Tipe 1': {
                        'dates': data['dates'],
                        'values': data['bbm_tipe_1']
                    },
                    'BBM Tipe 2': {
                        'dates': data['dates'],
                        'values': data['bbm_tipe_2']
                    }
                }
                
                fig_loc = self.viz_engine.create_comparison_chart(
                    loc_chart_data,
                    chart_type='line',
                    title=f"BBM Consumption - {loc}"
                )
                
                fig_loc.update_layout(yaxis_title="Consumption (Liters)")
                st.plotly_chart(fig_loc, use_container_width=True)
    
    def _render_summary_statistics(self):
        """Render summary statistics"""
        stats_data = []
        
        for loc, data in st.session_state.bbm_data.items():
            for bbm_type in self.bbm_types:
                if bbm_type == 'BBM Tipe 1':
                    values = data['bbm_tipe_1']
                else:
                    values = data['bbm_tipe_2']
                
                stats_data.append({
                    'Location': loc,
                    'BBM Type': bbm_type,
                    'Mean': f"{np.mean(values):,.0f} L",
                    'Std Dev': f"{np.std(values):,.0f} L",
                    'Min': f"{np.min(values):,.0f} L",
                    'Max': f"{np.max(values):,.0f} L",
                    'CV (%)': f"{(np.std(values) / np.mean(values)) * 100:.1f}%",
                    'Total': f"{np.sum(values):,.0f} L"
                })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Additional insights
        st.subheader("📈 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Highest consuming location
            location_totals = {}
            for loc, data in st.session_state.bbm_data.items():
                total = sum(data['bbm_tipe_1']) + sum(data['bbm_tipe_2'])
                location_totals[loc] = total
            
            highest_loc = max(location_totals, key=location_totals.get)
            st.info(f"🏆 **Highest Consumption**: {highest_loc} ({location_totals[highest_loc]:,.0f} L total)")
        
        with col2:
            # Most variable consumption
            cv_values = {}
            for loc, data in st.session_state.bbm_data.items():
                combined_values = data['bbm_tipe_1'] + data['bbm_tipe_2']
                cv = (np.std(combined_values) / np.mean(combined_values)) * 100
                cv_values[loc] = cv
            
            most_variable_loc = max(cv_values, key=cv_values.get)
            st.info(f"📊 **Most Variable**: {most_variable_loc} (CV: {cv_values[most_variable_loc]:.1f}%)")
    
    def render_analysis_results(self):
        """Render ARIMA analysis results"""
        if not st.session_state.analysis_completed:
            st.info("🔬 Run ARIMA analysis using the sidebar to see results here.")
            return
        
        st.header("🔬 ARIMA Analysis Results")
        
        # Results summary
        results = st.session_state.analysis_results
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
            st.metric("🔮 Forecast Months", st.session_state.forecast_periods)
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Forecast Charts", 
            "📊 Model Summary", 
            "📋 Detailed Results",
            "📥 Export Options"
        ])
        
        with tab1:
            self._render_forecast_charts()
        
        with tab2:
            self._render_model_summary()
        
        with tab3:
            self._render_detailed_results()
        
        with tab4:
            self._render_export_options()
    
    def _render_forecast_charts(self):
        """Render forecast visualization charts"""
        st.subheader("📈 ARIMA Forecast Visualizations")
        
        for (loc, bbm_type), result in st.session_state.analysis_results.items():
            
            st.write(f"### {loc} - {bbm_type}")
            
            if 'error' not in result:
                # Get historical data
                loc_data = st.session_state.bbm_data[loc]
                if bbm_type == 'BBM Tipe 1':
                    historical_data = loc_data['bbm_tipe_1']
                else:
                    historical_data = loc_data['bbm_tipe_2']
                
                dates = loc_data['dates']
                
                # Create forecast chart
                fig = self.viz_engine.create_forecast_chart(
                    historical_data=historical_data,
                    forecast_result=result,
                    dates=dates,
                    title=f"ARIMA Forecast - {loc} ({bbm_type})",
                    y_axis_title="Consumption (Liters)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show model metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # Handle different result formats
                if 'model_selection' in result:
                    # New format
                    arima_order = result['model_selection']['selected_order']
                    aic = result['model_fit']['aic']
                    rmse = result['validation'].get('rmse', None)
                    mape = result['validation'].get('mape', None)
                else:
                    # Legacy format
                    arima_order = result.get('arima_order', 'N/A')
                    aic = result.get('aic', 0)
                    rmse = result.get('rmse', None)
                    mape = result.get('mape', None)
                
                with col1:
                    st.metric("ARIMA Order", str(arima_order))
                with col2:
                    st.metric("AIC", f"{aic:.2f}" if aic else "N/A")
                with col3:
                    st.metric("RMSE", f"{rmse:.0f}" if rmse else "N/A")
                with col4:
                    st.metric("MAPE", f"{mape:.1f}%" if mape else "N/A")
                
            else:
                st.error(f"❌ Analysis failed: {result['error']}")
            
            st.markdown("---")
    
    def _render_model_summary(self):
        """Render model summary table"""
        st.subheader("📊 Model Performance Summary")
        
        # Create summary table
        summary_df = self.viz_engine.create_summary_table(
            st.session_state.analysis_results, 
            table_type='model_summary'
        )
        
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
            
            # Summary insights
            col1, col2 = st.columns(2)
            
            with col1:
                # Best performing model
                if 'MAPE (%)' in summary_df.columns:
                    # Remove 'N/A' and convert to numeric
                    mape_values = summary_df['MAPE (%)'].replace('N/A', np.nan)
                    mape_numeric = pd.to_numeric(mape_values, errors='coerce')
                    
                    if not mape_numeric.isna().all():
                        best_idx = mape_numeric.idxmin()
                        best_model = summary_df.iloc[best_idx]
                        st.success(f"🏆 **Best Model**: {best_model['Location']} - {best_model['Data Type']} (MAPE: {best_model['MAPE (%)']})")
            
            with col2:
                # Stationarity summary
                stationary_count = (summary_df['Stationary'] == '✅ Yes').sum()
                total_count = len(summary_df)
                st.info(f"📊 **Stationarity**: {stationary_count}/{total_count} series are stationary")
        
        else:
            st.warning("No valid analysis results to display.")
    
    def _render_detailed_results(self):
        """Render detailed analysis results"""
        st.subheader("📋 Detailed Forecast Results")
        
        # Location selector
        available_locations = list(set([loc for loc, _ in st.session_state.analysis_results.keys()]))
        selected_location = st.selectbox(
            "Select location for detailed view:",
            available_locations,
            help="Choose a location to view detailed forecast results"
        )
        
        # Show results for selected location
        for bbm_type in self.bbm_types:
            if (selected_location, bbm_type) in st.session_state.analysis_results:
                result = st.session_state.analysis_results[(selected_location, bbm_type)]
                
                st.write(f"#### {bbm_type}")
                
                if 'error' not in result:
                    # Get dates for forecast table
                    dates = st.session_state.bbm_data[selected_location]['dates']
                    
                    # Create forecast table
                    forecast_df = self.viz_engine.create_forecast_table(result, dates)
                    
                    if not forecast_df.empty:
                        st.dataframe(forecast_df, use_container_width=True)
                        
                        # Download forecast data
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {bbm_type} Forecast",
                            data=csv,
                            file_name=f"{selected_location}_{bbm_type.replace(' ', '_')}_forecast.csv",
                            mime="text/csv",
                            key=f"download_{selected_location}_{bbm_type}"
                        )
                    else:
                        st.warning(f"No forecast data available for {bbm_type}")
                else:
                    st.error(f"Analysis failed for {bbm_type}: {result['error']}")
                
                st.markdown("---")
    
    def _render_export_options(self):
        """Render export options for analysis results"""
        st.subheader("📥 Export Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Data Exports**")
            
            # Export all forecast data
            if st.button("📈 Export All Forecasts (CSV)", use_container_width=True):
                self._export_all_forecasts()
            
            # Export model summary
            if st.button("📋 Export Model Summary (CSV)", use_container_width=True):
                self._export_model_summary()
            
            # Export raw data
            if st.button("🗃️ Export Raw Data (CSV)", use_container_width=True):
                self._export_raw_data()
        
        with col2:
            st.write("**📈 Chart Exports**")
            
            # Export all charts
            if st.button("🖼️ Export All Charts (HTML)", use_container_width=True):
                self._export_all_charts()
            
            # Export analysis report
            if st.button("📄 Generate Analysis Report", use_container_width=True):
                self._generate_analysis_report()
        
        # Export format options
        st.markdown("---")
        st.write("**⚙️ Export Settings**")
        
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox(
                "Chart export format:",
                options=['HTML', 'PNG', 'SVG', 'PDF'],
                index=0,
                help="Choose format for chart exports"
            )
        
        with col2:
            include_confidence_intervals = st.checkbox(
                "Include confidence intervals",
                value=True,
                help="Include confidence intervals in forecast exports"
            )
    
    def _export_all_forecasts(self):
        """Export all forecast results to CSV"""
        all_forecasts = []
        
        for (loc, bbm_type), result in st.session_state.analysis_results.items():
            if 'error' not in result:
                dates = st.session_state.bbm_data[loc]['dates']
                forecast_df = self.viz_engine.create_forecast_table(result, dates)
                
                if not forecast_df.empty:
                    forecast_df['Location'] = loc
                    forecast_df['BBM_Type'] = bbm_type
                    all_forecasts.append(forecast_df)
        
        if all_forecasts:
            combined_df = pd.concat(all_forecasts, ignore_index=True)
            csv = combined_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download All Forecasts",
                data=csv,
                file_name=f"bbm_forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.success("✅ All forecasts prepared for download!")
        else:
            st.error("❌ No forecast data available for export.")
    
    def _export_model_summary(self):
        """Export model summary to CSV"""
        summary_df = self.viz_engine.create_summary_table(
            st.session_state.analysis_results, 
            table_type='model_summary'
        )
        
        if not summary_df.empty:
            csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Model Summary",
                data=csv,
                file_name=f"bbm_model_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.success("✅ Model summary prepared for download!")
        else:
            st.error("❌ No model summary available for export.")
    
    def _export_raw_data(self):
        """Export raw BBM data to CSV"""
        if st.session_state.bbm_data:
            # Prepare raw data
            raw_data = []
            
            for loc, data in st.session_state.bbm_data.items():
                for i, date in enumerate(data['dates']):
                    raw_data.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'Location': loc,
                        'BBM_Tipe_1': data['bbm_tipe_1'][i],
                        'BBM_Tipe_2': data['bbm_tipe_2'][i],
                        'Total_Consumption': data['bbm_tipe_1'][i] + data['bbm_tipe_2'][i]
                    })
            
            df = pd.DataFrame(raw_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Raw Data",
                data=csv,
                file_name=f"bbm_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.success("✅ Raw data prepared for download!")
        else:
            st.error("❌ No raw data available for export.")
    
    def _export_all_charts(self):
        """Export all charts as HTML bundle"""
        if st.session_state.analysis_results:
            st.info("🔄 Generating chart bundle... This may take a moment.")
            
            # This would generate individual chart files
            # For simplicity, we'll show the concept
            chart_count = len([r for r in st.session_state.analysis_results.values() if 'error' not in r])
            
            st.success(f"✅ Ready to export {chart_count} forecast charts!")
            st.info("💡 Individual chart export functionality can be implemented based on specific requirements.")
        else:
            st.error("❌ No charts available for export.")
    
    def _generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        if st.session_state.analysis_results:
            
            # Generate report content
            report_content = self._create_report_content()
            
            st.download_button(
                label="📄 Download Analysis Report",
                data=report_content,
                file_name=f"bbm_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
            st.success("✅ Analysis report generated!")
        else:
            st.error("❌ No analysis results available for report generation.")
    
    def _create_report_content(self) -> str:
        """Create markdown report content"""
        report = f"""# BBM Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### Analysis Overview
- **Total Locations Analyzed**: {len(set([loc for loc, _ in st.session_state.analysis_results.keys()]))}
- **BBM Types**: {len(self.bbm_types)}
- **Historical Data Period**: {len(list(st.session_state.bbm_data.values())[0]['dates'])} months
- **Forecast Period**: {st.session_state.forecast_periods} months

### Key Findings
"""
        
        # Add model performance summary
        successful_analyses = len([r for r in st.session_state.analysis_results.values() if 'error' not in r])
        total_analyses = len(st.session_state.analysis_results)
        
        report += f"""
#### Model Performance
- **Success Rate**: {(successful_analyses/total_analyses)*100:.1f}% ({successful_analyses}/{total_analyses})
- **Average AIC**: {self._calculate_average_aic():.2f}
- **Best Performing Model**: {self._get_best_model_info()}

#### Stationarity Analysis
"""
        
        # Add stationarity summary
        stationary_count = self._count_stationary_series()
        report += f"- **Stationary Series**: {stationary_count}/{total_analyses} ({(stationary_count/total_analyses)*100:.1f}%)\n"
        
        # Add detailed results for each location
        report += "\n## Detailed Results by Location\n"
        
        for loc in set([loc for loc, _ in st.session_state.analysis_results.keys()]):
            report += f"\n### {loc}\n"
            
            for bbm_type in self.bbm_types:
                if (loc, bbm_type) in st.session_state.analysis_results:
                    result = st.session_state.analysis_results[(loc, bbm_type)]
                    
                    if 'error' not in result:
                        # Handle different result formats
                        if 'model_selection' in result:
                            order = result['model_selection']['selected_order']
                            aic = result['model_fit']['aic']
                            rmse = result['validation'].get('rmse', 'N/A')
                            mape = result['validation'].get('mape', 'N/A')
                        else:
                            order = result.get('arima_order', 'N/A')
                            aic = result.get('aic', 'N/A')
                            rmse = result.get('rmse', 'N/A')
                            mape = result.get('mape', 'N/A')
                        
                        report += f"""
#### {bbm_type}
- **ARIMA Order**: {order}
- **AIC**: {aic:.2f if isinstance(aic, (int, float)) else aic}
- **RMSE**: {rmse:.2f if isinstance(rmse, (int, float)) else rmse}
- **MAPE**: {mape:.1f}% if isinstance(mape, (int, float)) else {mape}
"""
                    else:
                        report += f"\n#### {bbm_type}\n- **Status**: Analysis failed - {result['error']}\n"
        
        report += f"""
## Methodology

### Data Generation
- Seasonal patterns with configurable amplitude
- Trend components with growth rates
- Realistic noise simulation
- Location-specific parameters

### ARIMA Analysis
- Stationarity testing using Augmented Dickey-Fuller test
- Grid search for optimal (p,d,q) parameters
- Model validation using train-test split
- Forecast generation with confidence intervals

### Quality Assessment
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Akaike Information Criterion (AIC)
- Model diagnostic tests

---
*Report generated by BBM Analysis Dashboard*
"""
        
        return report
    
    def _calculate_average_aic(self) -> float:
        """Calculate average AIC across all successful analyses"""
        aic_values = []
        
        for result in st.session_state.analysis_results.values():
            if 'error' not in result:
                if 'model_fit' in result:
                    aic = result['model_fit']['aic']
                else:
                    aic = result.get('aic', None)
                
                if aic is not None:
                    aic_values.append(aic)
        
        return np.mean(aic_values) if aic_values else 0.0
    
    def _get_best_model_info(self) -> str:
        """Get information about the best performing model"""
        best_mape = float('inf')
        best_model = None
        
        for (loc, bbm_type), result in st.session_state.analysis_results.items():
            if 'error' not in result:
                if 'validation' in result:
                    mape = result['validation'].get('mape', None)
                else:
                    mape = result.get('mape', None)
                
                if mape is not None and mape < best_mape:
                    best_mape = mape
                    best_model = f"{loc} - {bbm_type}"
        
        return f"{best_model} (MAPE: {best_mape:.1f}%)" if best_model else "N/A"
    
    def _count_stationary_series(self) -> int:
        """Count number of stationary time series"""
        stationary_count = 0
        
        for result in st.session_state.analysis_results.values():
            if 'error' not in result:
                if 'stationarity' in result:
                    is_stationary = result['stationarity']['overall']['is_stationary']
                else:
                    is_stationary = result.get('is_stationary', False)
                
                if is_stationary:
                    stationary_count += 1
        
        return stationary_count
    
    def render_help_section(self):
        """Render help and documentation section"""
        with st.expander("❓ Help & Documentation", expanded=False):
            st.markdown("""
            ### 🎯 How to Use BBM Analysis Dashboard
            
            **Step 1: Configuration**
            - Set up locations and time periods in the sidebar
            - Configure BBM consumption parameters for each location
            - Adjust ARIMA model parameters if needed
            
            **Step 2: Data Generation**
            - Click "Generate Data" to create realistic BBM consumption data
            - Review the data preview to ensure parameters are correct
            - Data includes seasonal patterns and realistic variations
            
            **Step 3: ARIMA Analysis**
            - Click "Run ARIMA Analysis" to perform time series forecasting
            - The system will automatically find optimal ARIMA parameters
            - View progress and results in real-time
            
            **Step 4: Results Analysis**
            - Explore forecast charts with confidence intervals
            - Review model performance metrics
            - Export results and generate reports
            
            ### 📊 Understanding the Metrics
            
            **ARIMA Order (p,d,q)**
            - p: Number of autoregressive terms
            - d: Number of differencing operations
            - q: Number of moving average terms
            
            **Performance Metrics**
            - **AIC**: Akaike Information Criterion (lower is better)
            - **RMSE**: Root Mean Square Error (lower is better)
            - **MAPE**: Mean Absolute Percentage Error (lower is better)
            
            **Quality Ratings**
            - Excellent: MAPE < 5%
            - Good: MAPE 5-10%
            - Acceptable: MAPE 10-20%
            - Poor: MAPE > 20%
            
            ### 🔧 Troubleshooting
            
            **Common Issues:**
            - Insufficient data: Ensure at least 8 months of historical data
            - High variation: Reduce variation percentages for more stable results
            - Analysis failures: Try different ARIMA parameter ranges
            
            **Tips for Better Results:**
            - Use realistic consumption values based on actual data
            - Consider seasonal patterns in your location
            - Validate results against known consumption patterns
            """)

def main():
    """Main function to run BBM Analysis Dashboard"""
    
    # Initialize dashboard
    dashboard = BBMAnalysisDashboard()
    
    # Render page header
    dashboard.render_page_header()
    
    # Render configuration sidebar and get config
    config = dashboard.render_configuration_sidebar()
    
    # Handle data generation
    if config['generate_data']:
        dashboard.generate_bbm_data(config)
    
    # Handle analysis execution
    if config['run_analysis']:
        dashboard.run_arima_analysis(config)
    
    # Main content area
    if st.session_state.analysis_completed:
        # Show analysis results if completed
        dashboard.render_analysis_results()
    elif st.session_state.bbm_data:
        # Show data preview if data is generated
        dashboard.render_data_preview()
    else:
        # Show welcome and help section
        st.info("👈 **Getting Started**: Configure your analysis parameters in the sidebar and click 'Generate Data' to begin.")
        dashboard.render_help_section()

if __name__ == "__main__":
    main()