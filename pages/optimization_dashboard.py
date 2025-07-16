"""
MARLOFIR-P Optimization Dashboard
Interactive Streamlit dashboard for BBM distribution optimization
Combines ARIMA forecasting with Genetic Algorithm optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
try:
    from algorithms.optimization_engine import OptimizationEngine, OptimizationObjectives, OptimizationScenario
    from algorithms.genetic_algorithm import GAParameters
    from models.distribution_model import create_jakarta_bbm_network, LocationType
    from core.arima_engine import ARIMAEngine
    from core.visualization import create_forecast_chart, create_comparison_chart, create_metrics_table
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

def initialize_session_state():
    """Initialize session state variables"""
    if 'optimization_engine' not in st.session_state:
        st.session_state.optimization_engine = None
    if 'arima_results' not in st.session_state:
        st.session_state.arima_results = {}
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = {}
    if 'selected_scenarios' not in st.session_state:
        st.session_state.selected_scenarios = ['balanced']
    if 'network_initialized' not in st.session_state:
        st.session_state.network_initialized = False

def setup_optimization_engine():
    """Setup optimization engine with network"""
    if not st.session_state.network_initialized:
        with st.spinner("Initializing distribution network..."):
            network = create_jakarta_bbm_network()
            st.session_state.optimization_engine = OptimizationEngine(network)
            st.session_state.optimization_engine.load_predefined_scenarios()
            st.session_state.network_initialized = True
        st.success("✅ Distribution network initialized!")

def render_header():
    """Render dashboard header"""
    st.title("🔬 MARLOFIR-P Optimization Dashboard")
    st.markdown("""
    **Multi-Objective BBM Distribution Optimization with ARIMA Forecasting**
    
    This dashboard combines:
    - 📈 **ARIMA Time Series Forecasting** for demand prediction
    - 🧬 **Genetic Algorithm Optimization** for route planning
    - 🎯 **Multi-Objective Strategies** for different business goals
    """)
    
    st.divider()

def render_network_overview():
    """Render network overview section"""
    if st.session_state.optimization_engine:
        network_summary = st.session_state.optimization_engine.network.get_network_summary()
        
        # Network metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📍 Total Locations", 
                network_summary['total_locations'],
                help="Total number of locations in the network"
            )
        
        with col2:
            st.metric(
                "⛽ SPBU Count", 
                network_summary['spbu_count'],
                help="Number of gas stations (SPBU)"
            )
        
        with col3:
            st.metric(
                "🚛 Vehicle Fleet", 
                network_summary['vehicle_count'],
                help="Total number of vehicles available"
            )
        
        with col4:
            st.metric(
                "📊 Stock Ratio", 
                f"{network_summary['stock_ratio']:.1%}",
                help="Current stock level across all locations"
            )
        
        # Detailed network info
        with st.expander("📋 Detailed Network Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📦 Capacity & Demand")
                st.write(f"**Total Storage Capacity:** {network_summary['total_storage_capacity']:,.0f} L")
                st.write(f"**Daily Demand:** {network_summary['total_daily_demand']:,.0f} L")
                st.write(f"**Current Stock:** {network_summary['total_current_stock']:,.0f} L")
                st.write(f"**Coverage Days:** {network_summary['demand_coverage_days']:.1f} days")
            
            with col2:
                st.subheader("🚛 Fleet Information")
                st.write(f"**Fleet Capacity:** {network_summary['fleet_capacity']:,.0f} L")
                st.write(f"**Route Count:** {network_summary['route_count']:,}")
                
                # Location urgency
                locations = st.session_state.optimization_engine.network.locations
                urgency_data = []
                for loc_id, location in locations.items():
                    if location.type == LocationType.SPBU:
                        urgency = st.session_state.optimization_engine.network.calculate_demand_urgency(loc_id)
                        urgency_data.append({
                            'Location': location.name,
                            'Urgency': urgency,
                            'Stock Ratio': location.current_stock / location.capacity if location.capacity > 0 else 0
                        })
                
                if urgency_data:
                    urgency_df = pd.DataFrame(urgency_data)
                    st.subheader("🚨 Location Urgency")
                    st.dataframe(urgency_df, use_container_width=True)

def render_arima_input_section():
    """Render ARIMA input and forecasting section"""
    st.header("📈 ARIMA Forecasting Input")
    
    # Option to use sample data or upload
    input_method = st.radio(
        "Choose input method:",
        ["Generate Sample Data", "Use Previous ARIMA Results", "Manual Input"],
        horizontal=True
    )
    
    if input_method == "Generate Sample Data":
        if st.button("🎲 Generate Sample ARIMA Data", type="primary"):
            with st.spinner("Generating sample forecasting data..."):
                generate_sample_arima_data()
            st.success("✅ Sample ARIMA data generated!")
    
    elif input_method == "Use Previous ARIMA Results":
        if st.button("📥 Load from BBM Analysis", type="secondary"):
            load_arima_from_session()
    
    elif input_method == "Manual Input":
        render_manual_arima_input()
    
    # Display current ARIMA data
    if st.session_state.arima_results:
        st.subheader("📊 Current ARIMA Forecasts")
        
        # Summary table
        forecast_summary = []
        for location_id, data in st.session_state.arima_results.items():
            if isinstance(data, dict) and 'forecast' in data:
                forecast_values = data['forecast']
                forecast_summary.append({
                    'Location': location_id,
                    'Forecast Days': len(forecast_values),
                    'Daily Avg (L)': f"{np.mean(forecast_values):,.0f}",
                    'Weekly Total (L)': f"{np.sum(forecast_values):,.0f}",
                    'Min Daily (L)': f"{np.min(forecast_values):,.0f}",
                    'Max Daily (L)': f"{np.max(forecast_values):,.0f}"
                })
        
        if forecast_summary:
            forecast_df = pd.DataFrame(forecast_summary)
            st.dataframe(forecast_df, use_container_width=True)
            
            # Visualization
            create_forecast_visualization()

def generate_sample_arima_data():
    """Generate realistic sample ARIMA forecasting data"""
    if not st.session_state.optimization_engine:
        return
    
    locations = st.session_state.optimization_engine.network.locations
    spbu_locations = [loc_id for loc_id, loc in locations.items() if loc.type == LocationType.SPBU]
    
    sample_data = {}
    
    for location_id in spbu_locations:
        location = locations[location_id]
        base_demand = location.daily_demand
        
        # Generate 7-day forecast with trend and seasonality
        np.random.seed(hash(location_id) % 2147483647)  # Consistent but different per location
        
        # Base pattern with slight trend and weekend effect
        days = 7
        trend = np.linspace(0, base_demand * 0.1, days)  # Slight upward trend
        seasonal = np.sin(np.arange(days) * 2 * np.pi / 7) * base_demand * 0.2  # Weekly pattern
        noise = np.random.normal(0, base_demand * 0.1, days)  # Random variation
        
        forecast_values = base_demand + trend + seasonal + noise
        forecast_values = np.maximum(forecast_values, base_demand * 0.5)  # Minimum threshold
        
        # Generate confidence intervals
        uncertainty = base_demand * 0.15
        upper_bound = forecast_values + uncertainty
        lower_bound = forecast_values - uncertainty
        lower_bound = np.maximum(lower_bound, 0)
        
        sample_data[location_id] = {
            'forecast': forecast_values.tolist(),
            'confidence_intervals': {
                'upper': upper_bound.tolist(),
                'lower': lower_bound.tolist()
            },
            'metrics': {
                'mape': np.random.uniform(3, 8),
                'rmse': np.random.uniform(200, 500),
                'aic': np.random.uniform(150, 300)
            }
        }
    
    st.session_state.arima_results = sample_data

def load_arima_from_session():
    """Load ARIMA results from BBM Analysis session"""
    # This would integrate with the ARIMA results from the BBM Analysis page
    if 'bbm_arima_results' in st.session_state:
        st.session_state.arima_results = st.session_state.bbm_arima_results
        st.success("✅ ARIMA results loaded from BBM Analysis!")
    else:
        st.warning("⚠️ No ARIMA results found in BBM Analysis. Please run ARIMA analysis first.")

def render_manual_arima_input():
    """Render manual ARIMA input interface"""
    st.subheader("✏️ Manual ARIMA Input")
    
    if st.session_state.optimization_engine:
        locations = st.session_state.optimization_engine.network.locations
        spbu_locations = [(loc_id, loc.name) for loc_id, loc in locations.items() 
                         if loc.type == LocationType.SPBU]
        
        selected_location = st.selectbox(
            "Select location for forecast input:",
            options=[loc[0] for loc in spbu_locations],
            format_func=lambda x: next(name for loc_id, name in spbu_locations if loc_id == x)
        )
        
        if selected_location:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Daily Forecast Values (Liters):**")
                forecast_input = st.text_area(
                    "Enter 7 daily values (one per line):",
                    value="5000\n5200\n4800\n5100\n5300\n4900\n5050",
                    height=150
                )
            
            with col2:
                mape = st.number_input("MAPE (%)", value=5.0, min_value=0.0, max_value=100.0)
                rmse = st.number_input("RMSE", value=250.0, min_value=0.0)
                aic = st.number_input("AIC", value=200.0, min_value=0.0)
            
            if st.button("Add Forecast"):
                try:
                    forecast_values = [float(line.strip()) for line in forecast_input.split('\n') if line.strip()]
                    if len(forecast_values) != 7:
                        st.error("Please enter exactly 7 daily values")
                    else:
                        if selected_location not in st.session_state.arima_results:
                            st.session_state.arima_results[selected_location] = {}
                        
                        st.session_state.arima_results[selected_location] = {
                            'forecast': forecast_values,
                            'confidence_intervals': {
                                'upper': [v * 1.15 for v in forecast_values],
                                'lower': [v * 0.85 for v in forecast_values]
                            },
                            'metrics': {
                                'mape': mape,
                                'rmse': rmse,
                                'aic': aic
                            }
                        }
                        st.success(f"✅ Forecast added for {selected_location}")
                except ValueError:
                    st.error("Please enter valid numeric values")

def create_forecast_visualization():
    """Create forecast visualization charts"""
    if not st.session_state.arima_results:
        return
    
    st.subheader("📊 Forecast Visualization")
    
    # Select location for detailed view
    location_options = list(st.session_state.arima_results.keys())
    selected_loc = st.selectbox("Select location for detailed forecast:", location_options)
    
    if selected_loc and selected_loc in st.session_state.arima_results:
        forecast_data = st.session_state.arima_results[selected_loc]
        
        if 'forecast' in forecast_data:
            days = list(range(1, len(forecast_data['forecast']) + 1))
            
            fig = go.Figure()
            
            # Main forecast line
            fig.add_trace(go.Scatter(
                x=days,
                y=forecast_data['forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # Confidence intervals if available
            if 'confidence_intervals' in forecast_data:
                ci = forecast_data['confidence_intervals']
                if 'upper' in ci and 'lower' in ci:
                    # Upper bound
                    fig.add_trace(go.Scatter(
                        x=days,
                        y=ci['upper'],
                        mode='lines',
                        name='Upper CI',
                        line=dict(color='rgba(255,0,0,0.3)', dash='dash'),
                        showlegend=False
                    ))
                    
                    # Lower bound
                    fig.add_trace(go.Scatter(
                        x=days,
                        y=ci['lower'],
                        mode='lines',
                        name='Lower CI',
                        line=dict(color='rgba(255,0,0,0.3)', dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)',
                        showlegend=False
                    ))
            
            fig.update_layout(
                title=f"📈 Demand Forecast - {selected_loc}",
                xaxis_title="Day",
                yaxis_title="Demand (Liters)",
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Forecast metrics
            if 'metrics' in forecast_data:
                metrics = forecast_data['metrics']
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
                with col2:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.1f}")
                with col3:
                    st.metric("AIC", f"{metrics.get('aic', 0):.1f}")

def render_optimization_configuration():
    """Render optimization configuration section"""
    st.header("⚙️ Optimization Configuration")
    
    # Scenario selection
    st.subheader("🎯 Select Optimization Scenarios")
    
    if st.session_state.optimization_engine:
        available_scenarios = st.session_state.optimization_engine.scenarios
        
        scenario_descriptions = {
            'cost_focused': '💰 Minimize total distribution costs',
            'time_focused': '⏰ Minimize delivery time',
            'balanced': '⚖️ Balance all objectives equally',
            'service_focused': '🎯 Maximize service level',
            'eco_friendly': '🌱 Minimize environmental impact'
        }
        
        # Multi-select for scenarios
        selected_scenarios = st.multiselect(
            "Choose optimization scenarios to run:",
            options=list(available_scenarios.keys()),
            default=st.session_state.selected_scenarios,
            format_func=lambda x: scenario_descriptions.get(x, x)
        )
        
        st.session_state.selected_scenarios = selected_scenarios
        
        # Advanced configuration
        with st.expander("🔧 Advanced Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                complexity_factor = st.slider(
                    "Problem Complexity Factor",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Adjusts GA parameters based on problem difficulty"
                )
                
                max_time = st.number_input(
                    "Max Optimization Time (minutes)",
                    min_value=1,
                    max_value=60,
                    value=10,
                    help="Maximum time allowed per scenario"
                )
            
            with col2:
                auto_tuning = st.checkbox(
                    "Auto Parameter Tuning",
                    value=True,
                    help="Automatically adjust GA parameters"
                )
                
                save_results = st.checkbox(
                    "Save Intermediate Results",
                    value=True,
                    help="Save results during optimization"
                )
            
            # Update engine configuration
            if st.session_state.optimization_engine:
                config_update = {
                    'max_optimization_time_minutes': max_time,
                    'auto_parameter_tuning': auto_tuning,
                    'save_intermediate_results': save_results
                }
                st.session_state.optimization_engine.set_configuration(config_update)
        
        # Custom scenario creation
        with st.expander("🎨 Create Custom Scenario"):
            render_custom_scenario_creator()

def render_custom_scenario_creator():
    """Render custom scenario creation interface"""
    st.subheader("Create Custom Optimization Scenario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scenario_name = st.text_input("Scenario Name", value="Custom Scenario")
        scenario_desc = st.text_area("Description", value="Custom optimization scenario")
    
    with col2:
        population_size = st.number_input("Population Size", min_value=20, max_value=200, value=100)
        generations = st.number_input("Generations", min_value=20, max_value=300, value=150)
    
    st.subheader("Objective Weights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cost_weight = st.slider("Cost Weight", 0.0, 1.0, 0.25, 0.05)
        time_weight = st.slider("Time Weight", 0.0, 1.0, 0.25, 0.05)
    
    with col2:
        service_weight = st.slider("Service Weight", 0.0, 1.0, 0.25, 0.05)
        efficiency_weight = st.slider("Fuel Efficiency Weight", 0.0, 1.0, 0.15, 0.05)
    
    with col3:
        distance_weight = st.slider("Distance Weight", 0.0, 1.0, 0.1, 0.05)
        carbon_weight = st.slider("Carbon Weight", 0.0, 1.0, 0.0, 0.05)
    
    # Normalize weights
    total_weight = cost_weight + time_weight + service_weight + efficiency_weight + distance_weight + carbon_weight
    if total_weight > 0:
        st.info(f"Total weight: {total_weight:.2f} (will be normalized to 1.0)")
    
    if st.button("Create Custom Scenario"):
        custom_objectives = OptimizationObjectives(
            cost_weight=cost_weight,
            time_weight=time_weight,
            service_level_weight=service_weight,
            fuel_efficiency_weight=efficiency_weight,
            distance_weight=distance_weight,
            carbon_footprint_weight=carbon_weight
        )
        
        custom_params = GAParameters(
            population_size=population_size,
            generations=generations
        )
        
        custom_scenario = OptimizationScenario(
            scenario_id=f"custom_{int(time.time())}",
            name=scenario_name,
            description=scenario_desc,
            objectives=custom_objectives,
            ga_parameters=custom_params
        )
        
        st.session_state.optimization_engine.add_scenario(custom_scenario)
        st.success(f"✅ Custom scenario '{scenario_name}' created!")

def render_optimization_execution():
    """Render optimization execution section"""
    st.header("🚀 Run Optimization")
    
    # Validation checks
    can_optimize = True
    issues = []
    
    if not st.session_state.arima_results:
        issues.append("❌ No ARIMA forecast data available")
        can_optimize = False
    
    if not st.session_state.selected_scenarios:
        issues.append("❌ No optimization scenarios selected")
        can_optimize = False
    
    if not st.session_state.optimization_engine:
        issues.append("❌ Optimization engine not initialized")
        can_optimize = False
    
    # Display issues if any
    if issues:
        st.error("Please resolve the following issues before optimization:")
        for issue in issues:
            st.write(issue)
    
    # Optimization execution
    if can_optimize:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔬 Run Single Scenario", type="secondary"):
                if len(st.session_state.selected_scenarios) > 0:
                    run_single_optimization(st.session_state.selected_scenarios[0])
        
        with col2:
            if st.button("🔬 Run All Scenarios", type="primary"):
                run_multiple_optimization()
        
        with col3:
            if st.button("🔄 Clear Results"):
                st.session_state.optimization_results = {}
                st.rerun()

def run_single_optimization(scenario_id: str):
    """Run single scenario optimization"""
    with st.spinner(f"Running optimization for scenario: {scenario_id}..."):
        try:
            start_time = time.time()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress
            progress_bar.progress(0.2)
            status_text.text("Processing ARIMA data...")
            
            # Run optimization
            result = st.session_state.optimization_engine.optimize_scenario(
                scenario_id, 
                st.session_state.arima_results,
                complexity_factor=1.0
            )
            
            progress_bar.progress(0.8)
            status_text.text("Processing results...")
            
            # Store results
            st.session_state.optimization_results[scenario_id] = result
            
            progress_bar.progress(1.0)
            status_text.text("Optimization completed!")
            
            optimization_time = time.time() - start_time
            st.success(f"✅ Optimization completed in {optimization_time:.1f} seconds!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")

def run_multiple_optimization():
    """Run multiple scenario optimization"""
    with st.spinner("Running multi-scenario optimization..."):
        try:
            start_time = time.time()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_scenarios = len(st.session_state.selected_scenarios)
            
            for i, scenario_id in enumerate(st.session_state.selected_scenarios):
                progress = (i + 1) / total_scenarios
                progress_bar.progress(progress)
                status_text.text(f"Optimizing scenario {i+1}/{total_scenarios}: {scenario_id}")
                
                # Run optimization
                result = st.session_state.optimization_engine.optimize_scenario(
                    scenario_id,
                    st.session_state.arima_results,
                    complexity_factor=1.0
                )
                
                st.session_state.optimization_results[scenario_id] = result
            
            progress_bar.progress(1.0)
            status_text.text("All optimizations completed!")
            
            optimization_time = time.time() - start_time
            st.success(f"✅ All {total_scenarios} scenarios completed in {optimization_time:.1f} seconds!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Multi-scenario optimization failed: {str(e)}")

def render_optimization_results():
    """Render optimization results section"""
    if not st.session_state.optimization_results:
        return
    
    st.header("📊 Optimization Results")
    
    # Results summary
    render_results_summary()
    
    # Detailed results
    render_detailed_results()
    
    # Scenario comparison
    if len(st.session_state.optimization_results) > 1:
        render_scenario_comparison()

def render_results_summary():
    """Render optimization results summary"""
    st.subheader("📈 Results Summary")
    
    # Create summary metrics
    summary_data = []
    for scenario_id, result in st.session_state.optimization_results.items():
        if 'optimization_summary' in result and 'route_optimization' in result:
            opt_summary = result['optimization_summary']
            route_opt = result['route_optimization']
            scenario_info = result.get('scenario_info', {})
            
            summary_data.append({
                'Scenario': scenario_info.get('scenario_name', scenario_id),
                'Fitness Score': f"{opt_summary['optimization_fitness']:.2f}",
                'Total Cost (IDR)': f"{route_opt['total_cost']:,.0f}",
                'Distance (km)': f"{route_opt['total_distance']:.1f}",
                'Service Level': f"{opt_summary['service_level_achieved']:.1%}",
                'Optimization Time (s)': f"{scenario_info.get('optimization_time_seconds', 0):.1f}"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

def render_detailed_results():
    """Render detailed results for selected scenario"""
    st.subheader("🔍 Detailed Results")
    
    # Scenario selector
    scenario_options = list(st.session_state.optimization_results.keys())
    selected_scenario = st.selectbox(
        "Select scenario for detailed view:",
        scenario_options,
        format_func=lambda x: st.session_state.optimization_results[x].get('scenario_info', {}).get('scenario_name', x)
    )
    
    if selected_scenario and selected_scenario in st.session_state.optimization_results:
        result = st.session_state.optimization_results[selected_scenario]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        opt_summary = result.get('optimization_summary', {})
        route_opt = result.get('route_optimization', {})
        
        with col1:
            st.metric(
                "🎯 Fitness Score",
                f"{opt_summary.get('optimization_fitness', 0):.2f}",
                help="Overall optimization performance score"
            )
        
        with col2:
            st.metric(
                "💰 Total Cost",
                f"IDR {route_opt.get('total_cost', 0):,.0f}",
                help="Total distribution cost"
            )
        
        with col3:
            st.metric(
                "📏 Total Distance",
                f"{route_opt.get('total_distance', 0):.1f} km",
                help="Total route distance"
            )
        
        with col4:
            st.metric(
                "⭐ Service Level",
                f"{opt_summary.get('service_level_achieved', 0):.1%}",
                help="Customer service level achieved"
            )
        
        # Route visualization
        create_route_visualization(result)
        
        # Convergence chart
        create_convergence_chart(result)
        
        # Recommendations
        if 'recommendations' in result and result['recommendations']:
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(result['recommendations'], 1):
                st.write(f"{i}. {rec}")

def create_route_visualization(result: Dict):
    """Create route visualization"""
    st.subheader("🗺️ Optimized Route")
    
    route_opt = result.get('route_optimization', {})
    route_sequence = route_opt.get('route_sequence', [])
    
    if route_sequence:
        # Create route visualization
        fig = go.Figure()
        
        # Add route points
        for i, location in enumerate(route_sequence):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                text=location,
                textposition='top center',
                name=f"Stop {i+1}",
                marker=dict(size=15, color=f'hsl({i*40}, 70%, 50%)')
            ))
        
        # Add route line
        fig.add_trace(go.Scatter(
            x=list(range(len(route_sequence))),
            y=[0] * len(route_sequence),
            mode='lines',
            name='Route',
            line=dict(color='gray', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title="📍 Optimized Delivery Route",
            xaxis_title="Route Sequence",
            yaxis=dict(visible=False),
            showlegend=False,
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Route details table
        location_analysis = result.get('location_analysis', [])
        if location_analysis:
            st.subheader("📋 Location Details")
            
            location_df = pd.DataFrame(location_analysis)
            # Format numeric columns
            if 'forecasted_demand' in location_df.columns:
                location_df['forecasted_demand'] = location_df['forecasted_demand'].apply(lambda x: f"{x:,.0f}")
            if 'current_stock' in location_df.columns:
                location_df['current_stock'] = location_df['current_stock'].apply(lambda x: f"{x:,.0f}")
            if 'urgency_score' in location_df.columns:
                location_df['urgency_score'] = location_df['urgency_score'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(location_df, use_container_width=True)

def create_convergence_chart(result: Dict):
    """Create GA convergence visualization"""
    convergence_history = result.get('convergence_history', [])
    
    if convergence_history:
        st.subheader("📈 Algorithm Convergence")
        
        # Extract data for plotting
        generations = [gen['generation'] for gen in convergence_history]
        best_fitness = [gen['best_fitness'] for gen in convergence_history]
        avg_fitness = [gen['avg_fitness'] for gen in convergence_history]
        diversity = [gen['diversity'] for gen in convergence_history]
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Fitness Evolution', 'Population Diversity'),
            vertical_spacing=0.1
        )
        
        # Fitness plot
        fig.add_trace(
            go.Scatter(x=generations, y=best_fitness, name='Best Fitness', 
                      line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=generations, y=avg_fitness, name='Average Fitness',
                      line=dict(color='#ff7f0e', width=2)),
            row=1, col=1
        )
        
        # Diversity plot
        fig.add_trace(
            go.Scatter(x=generations, y=diversity, name='Diversity',
                      line=dict(color='#2ca02c', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Generation", row=2, col=1)
        fig.update_yaxes(title_text="Fitness Score", row=1, col=1)
        fig.update_yaxes(title_text="Diversity Score", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

def render_scenario_comparison():
    """Render scenario comparison section"""
    st.subheader("⚖️ Scenario Comparison")
    
    # Comparison metrics
    comparison_data = []
    for scenario_id, result in st.session_state.optimization_results.items():
        if 'optimization_summary' in result and 'route_optimization' in result:
            opt_summary = result['optimization_summary']
            route_opt = result['route_optimization']
            scenario_info = result.get('scenario_info', {})
            
            comparison_data.append({
                'Scenario': scenario_info.get('scenario_name', scenario_id),
                'Fitness': opt_summary['optimization_fitness'],
                'Cost': route_opt['total_cost'],
                'Distance': route_opt['total_distance'],
                'Service Level': opt_summary['service_level_achieved'],
                'Fuel Efficiency': route_opt.get('fuel_efficiency', 0)
            })
    
    if len(comparison_data) >= 2:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Multi-metric comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fitness Comparison', 'Cost Comparison', 
                          'Distance Comparison', 'Service Level Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        scenarios = comparison_df['Scenario']
        
        # Fitness comparison
        fig.add_trace(
            go.Bar(x=scenarios, y=comparison_df['Fitness'], name='Fitness',
                   marker_color='#1f77b4'),
            row=1, col=1
        )
        
        # Cost comparison
        fig.add_trace(
            go.Bar(x=scenarios, y=comparison_df['Cost'], name='Cost',
                   marker_color='#ff7f0e'),
            row=1, col=2
        )
        
        # Distance comparison
        fig.add_trace(
            go.Bar(x=scenarios, y=comparison_df['Distance'], name='Distance',
                   marker_color='#2ca02c'),
            row=2, col=1
        )
        
        # Service level comparison
        fig.add_trace(
            go.Bar(x=scenarios, y=comparison_df['Service Level'], name='Service Level',
                   marker_color='#d62728'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best scenario analysis
        st.subheader("🏆 Best Scenario Analysis")
        
        best_fitness = comparison_df.loc[comparison_df['Fitness'].idxmax(), 'Scenario']
        best_cost = comparison_df.loc[comparison_df['Cost'].idxmin(), 'Scenario']
        best_service = comparison_df.loc[comparison_df['Service Level'].idxmax(), 'Scenario']
        best_distance = comparison_df.loc[comparison_df['Distance'].idxmin(), 'Scenario']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Best Overall", best_fitness)
        with col2:
            st.metric("💰 Lowest Cost", best_cost)
        with col3:
            st.metric("⭐ Best Service", best_service)
        with col4:
            st.metric("📏 Shortest Distance", best_distance)

def render_export_section():
    """Render export and download section"""
    if not st.session_state.optimization_results:
        return
    
    st.header("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to Excel", type="secondary"):
            export_to_excel()
    
    with col2:
        if st.button("📋 Export Summary Report", type="secondary"):
            export_summary_report()
    
    with col3:
        if st.button("📈 Export Charts", type="secondary"):
            export_charts()

def export_to_excel():
    """Export results to Excel format"""
    try:
        # Create Excel data
        excel_data = {}
        
        # Summary sheet
        summary_data = []
        for scenario_id, result in st.session_state.optimization_results.items():
            if 'optimization_summary' in result:
                summary_data.append({
                    'Scenario': result.get('scenario_info', {}).get('scenario_name', scenario_id),
                    'Fitness': result['optimization_summary']['optimization_fitness'],
                    'Cost': result['route_optimization']['total_cost'],
                    'Distance': result['route_optimization']['total_distance'],
                    'Service Level': result['optimization_summary']['service_level_achieved']
                })
        
        excel_data['Summary'] = pd.DataFrame(summary_data)
        
        # Location details for each scenario
        for scenario_id, result in st.session_state.optimization_results.items():
            if 'location_analysis' in result:
                scenario_name = result.get('scenario_info', {}).get('scenario_name', scenario_id)
                excel_data[f'Locations_{scenario_name}'] = pd.DataFrame(result['location_analysis'])
        
        # Create download button with Excel data
        excel_buffer = create_excel_download(excel_data)
        
        st.download_button(
            label="📊 Download Excel Report",
            data=excel_buffer,
            file_name=f"marlofir_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.success("✅ Excel export ready for download!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def create_excel_download(excel_data: Dict[str, pd.DataFrame]) -> bytes:
    """Create Excel file in memory"""
    from io import BytesIO
    
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for sheet_name, df in excel_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    buffer.seek(0)
    return buffer.getvalue()

def export_summary_report():
    """Export summary report as JSON"""
    try:
        # Create comprehensive report
        report = {
            'export_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_scenarios': len(st.session_state.optimization_results),
                'network_summary': st.session_state.optimization_engine.network.get_network_summary() if st.session_state.optimization_engine else {}
            },
            'optimization_results': st.session_state.optimization_results,
            'arima_forecasts': st.session_state.arima_results
        }
        
        # Convert to JSON
        report_json = json.dumps(report, indent=2, default=str)
        
        st.download_button(
            label="📋 Download JSON Report",
            data=report_json,
            file_name=f"marlofir_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
        
        st.success("✅ JSON report ready for download!")
        
    except Exception as e:
        st.error(f"❌ Report export failed: {str(e)}")

def export_charts():
    """Export visualization charts"""
    st.info("📈 Chart export feature coming soon!")

def main():
    """Main dashboard function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Setup optimization engine
    setup_optimization_engine()
    
    # Render main sections
    render_header()
    
    # Main content in tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Network Overview", 
        "📈 ARIMA Input", 
        "⚙️ Optimization", 
        "📊 Results"
    ])
    
    with tab1:
        render_network_overview()
    
    with tab2:
        render_arima_input_section()
    
    with tab3:
        render_optimization_configuration()
        st.divider()
        render_optimization_execution()
    
    with tab4:
        render_optimization_results()
        st.divider()
        render_export_section()
    
    # Sidebar with quick info
    with st.sidebar:
        st.header("📊 Quick Stats")
        
        if st.session_state.optimization_engine:
            perf_metrics = st.session_state.optimization_engine.performance_metrics
            
            st.metric("Total Optimizations", perf_metrics['total_optimizations'])
            st.metric("Avg Time (s)", f"{perf_metrics['avg_optimization_time']:.1f}")
            st.metric("Best Fitness", f"{perf_metrics['best_fitness_achieved']:.2f}")
        
        st.divider()
        
        st.header("🔗 Quick Actions")
        
        if st.button("🎲 Generate Sample Data", key="sidebar_sample"):
            generate_sample_arima_data()
            st.success("Sample data generated!")
        
        if st.button("🔄 Reset All", key="sidebar_reset"):
            for key in ['arima_results', 'optimization_results', 'selected_scenarios']:
                if key in st.session_state:
                    st.session_state[key] = {} if 'results' in key else ['balanced']
            st.success("Reset complete!")
            st.rerun()
        
        st.divider()
        
        st.header("ℹ️ About MARLOFIR-P")
        st.markdown("""
        **MARLOFIR-P** combines:
        
        🧬 **Genetic Algorithm** optimization for route planning
        
        📈 **ARIMA Forecasting** for demand prediction
        
        🎯 **Multi-Objective** optimization strategies
        
        ⚖️ **Scenario Comparison** for decision support
        """)

if __name__ == "__main__":
    main()