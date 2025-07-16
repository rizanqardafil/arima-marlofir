"""
main.py - BBM Dashboard Entry Point (Production Ready)
Complete MARLOFIR-P system with Phase 1 & Phase 2 integration
"""

import streamlit as st
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="MARLOFIR-P BBM Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS - Clean UI without hiding functionality
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: bold;
    margin-bottom: 1rem;
}

.metric-card {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    transition: all 0.3s ease;
}

.metric-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.info-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #1976d2;
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
    # Header
    st.markdown('<h1 class="main-header">⛽ MARLOFIR-P BBM Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Complete BBM Distribution Optimization System**")
    st.markdown("---")
    
    # Navigation menu in sidebar
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("### Choose Analysis Module:")
    
    # Updated page selection with Phase 2
    page_options = {
        "🏠 Home": "Welcome & System Overview",
        "⛽ BBM Analysis": "ARIMA Demand Forecasting", 
        "🚗 Transport Mode Analysis": "Transport Mode BBM Analysis",
        "🔬 MARLOFIR-P Optimization": "Multi-Objective Route Optimization",
        "🌊 Wave Scheduling": "Maritime Delivery Scheduling",
        "📊 Comparison Dashboard": "Comparative Analysis Tools"
    }
    
    page = st.sidebar.selectbox(
        "Select Module:",
        list(page_options.keys()),
        format_func=lambda x: f"{x}",
        key="main_navigation"
    )
    
    # Show page description
    st.sidebar.markdown(f"*{page_options[page]}*")
    
    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 System Status:")
    
    system_modules = [
        ("Phase 1: ARIMA Dashboard", "✅", "Operational"),
        ("Phase 2: MARLOFIR-P", "✅", "Operational"),
        ("Phase 3: Wave Scheduling", "✅", "Operational")
    ]
    
    for module, status, desc in system_modules:
        st.sidebar.markdown(f"{status} **{module}**: {desc}")
    
    # Quick stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Capabilities:")
    
    sidebar_col1, sidebar_col2 = st.sidebar.columns(2)
    with sidebar_col1:
        st.metric("Analysis Types", "6", help="BBM, Transport, Routes, Waves")
        st.metric("Algorithms", "4", help="ARIMA, GA, VRP, Memetic")
    
    with sidebar_col2:
        st.metric("Max Locations", "10+", help="Scalable network size")
        st.metric("Export Formats", "8", help="Excel, PDF, JSON, etc.")
    
    # Route to appropriate page
    if page == "🏠 Home":
        render_home_page()
    elif page == "⛽ BBM Analysis":
        render_bbm_analysis_page()
    elif page == "🚗 Transport Mode Analysis":
        render_transport_analysis_page()
    elif page == "🔬 MARLOFIR-P Optimization":
        render_optimization_page()
    elif page == "🌊 Wave Scheduling":
        render_wave_scheduling_page()
    elif page == "📊 Comparison Dashboard":
        render_comparison_page()

def render_home_page():
    """Render enhanced home page with Phase 2 info"""
    
    st.header("🏠 MARLOFIR-P System Overview")
    
    # System introduction
    st.markdown("""
    <div class="info-box">
        <h3 style='color: #1976d2; margin-top: 0; font-weight: bold;'>🎯 Complete BBM Distribution Optimization Platform</h3>
        <p style='margin-bottom: 0; font-size: 1.1rem; color: #424242;'>
            Advanced multi-phase system combining ARIMA forecasting, genetic algorithm optimization, 
            and wave-influenced maritime scheduling for comprehensive BBM distribution management.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System phases
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="height: 300px;">
            <h4 style='color: #1f77b4;'>📈 Phase 1: Forecasting</h4>
            <ul style='color: #555;'>
                <li><strong>ARIMA Analysis</strong> - Demand forecasting</li>
                <li><strong>Time Series</strong> - Seasonal patterns</li>
                <li><strong>BBM & Transport</strong> - Multi-modal analysis</li>
                <li><strong>Data Processing</strong> - Quality assurance</li>
                <li><strong>Interactive Charts</strong> - Real-time visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="height: 300px;">
            <h4 style='color: #ff7f0e;'>🧬 Phase 2: Optimization</h4>
            <ul style='color: #555;'>
                <li><strong>Genetic Algorithm</strong> - Route optimization</li>
                <li><strong>VRP Solver</strong> - Vehicle routing problems</li>
                <li><strong>Multi-Objective</strong> - Cost, time, service</li>
                <li><strong>Geographic Analysis</strong> - Distance optimization</li>
                <li><strong>Professional Export</strong> - Business reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="height: 300px;">
            <h4 style='color: #2ca02c;'>🌊 Phase 3: Maritime</h4>
            <ul style='color: #555;'>
                <li><strong>Memetic Algorithm</strong> - Wave-aware scheduling</li>
                <li><strong>Oceanographic Data</strong> - Weather integration</li>
                <li><strong>Maritime Safety</strong> - Vessel operability</li>
                <li><strong>Tidal Analysis</strong> - Indonesian waters</li>
                <li><strong>Schedule Optimization</strong> - Weather windows</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # System capabilities
    st.markdown("---")
    st.subheader("🚀 Complete System Capabilities")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Analysis Modules", "6", delta="Integrated", help="All analysis types available")
    with col2:
        st.metric("Optimization Algorithms", "4", delta="Advanced", help="ARIMA, GA, VRP, Memetic")
    with col3:
        st.metric("Export Formats", "8", delta="Professional", help="Excel, PDF, JSON, Charts")
    with col4:
        st.metric("Integration Points", "17", delta="Complete", help="All modules integrated")
    
    # Feature matrix
    st.markdown("---")
    st.subheader("📋 Feature Matrix")
    
    feature_data = {
        "Module": ["BBM Analysis", "Transport Analysis", "Route Optimization", "Wave Scheduling", "Data Export", "Geographic Utils"],
        "Status": ["✅ Complete", "✅ Complete", "✅ Complete", "✅ Complete", "✅ Complete", "✅ Complete"],
        "Algorithms": ["ARIMA", "ARIMA + Stats", "GA + VRP", "Memetic + Ocean", "Multi-format", "Geographic + Maps"],
        "Use Case": ["Demand Forecast", "Modal Analysis", "Route Planning", "Maritime Safety", "Reporting", "Distance & Clustering"]
    }
    
    import pandas as pd
    feature_df = pd.DataFrame(feature_data)
    st.dataframe(feature_df, use_container_width=True)
    
    # Quick start guide
    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For Demand Forecasting:**
        1. Go to **BBM Analysis** or **Transport Analysis**
        2. Configure locations and time periods
        3. Generate historical data
        4. Run ARIMA forecasting
        5. Export results and charts
        """)
    
    with col2:
        st.markdown("""
        **For Route Optimization:**
        1. Go to **MARLOFIR-P Optimization**
        2. Input ARIMA forecast results
        3. Configure optimization scenarios
        4. Run genetic algorithm optimization
        5. Download optimization reports
        """)

def render_bbm_analysis_page():
    """BBM Analysis page - Phase 1"""
    try:
        from pages.bbm_analysis import BBMAnalysisDashboard
        dashboard = BBMAnalysisDashboard()
        dashboard.render_page_header()
        config = dashboard.render_configuration_sidebar()
        
        if config['generate_data']:
            dashboard.generate_bbm_data(config)
        if config['run_analysis']:
            dashboard.run_arima_analysis(config)
        
        if st.session_state.get('analysis_completed', False):
            dashboard.render_analysis_results()
        elif st.session_state.get('bbm_data', {}):
            dashboard.render_data_preview()
        else:
            st.info("👈 Configure parameters in sidebar to begin BBM analysis")
            dashboard.render_help_section()
            
    except ImportError as e:
        st.error(f"❌ BBM Analysis module not found: {e}")
        st.info("Ensure pages/bbm_analysis.py exists with required core modules")

def render_transport_analysis_page():
    """Transport Analysis page - Phase 1"""
    try:
        from pages.transport_analysis import TransportAnalysisDashboard
        dashboard = TransportAnalysisDashboard()
        dashboard.render_page_header()
        config = dashboard.render_configuration_sidebar()
        
        if config['generate_data']:
            dashboard.generate_transport_data(config)
        if config['run_analysis']:
            dashboard.run_transport_analysis(config)
        
        if st.session_state.get('transport_analysis_completed', False):
            dashboard.render_analysis_results()
        elif st.session_state.get('transport_data', {}):
            dashboard.render_data_preview()
        else:
            st.info("👈 Configure transport modes in sidebar to begin analysis")
            dashboard.render_help_section()
            
    except ImportError as e:
        st.error(f"❌ Transport Analysis module not found: {e}")
        st.info("Ensure pages/transport_analysis.py exists with required core modules")

def render_optimization_page():
    """MARLOFIR-P Optimization page - Phase 2"""
    try:
        from pages.optimization_dashboard import main as optimization_main
        optimization_main()
        
    except ImportError as e:
        st.error(f"❌ MARLOFIR-P Optimization module not found: {e}")
        st.markdown("""
        **Missing Phase 2 Modules:**
        - `pages/optimization_dashboard.py`
        - `algorithms/genetic_algorithm.py`
        - `algorithms/optimization_engine.py`
        - `models/distribution_model.py`
        - `models/vehicle_routing.py`
        - `core/integration_engine.py`
        """)
        
        # Show Phase 2 preview
        st.markdown("---")
        st.subheader("🔬 MARLOFIR-P Optimization Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Genetic Algorithm Features:**
            - Multi-objective optimization
            - Route planning for BBM distribution
            - Vehicle capacity constraints
            - Cost and time optimization
            - Convergence tracking
            """)
        
        with col2:
            st.markdown("""
            **Integration Capabilities:**
            - ARIMA forecast input
            - Multiple optimization scenarios
            - Professional report generation
            - Geographic route visualization
            - Performance benchmarking
            """)

def render_wave_scheduling_page():
    """Wave Scheduling page - Phase 3"""
    try:
        # Try to import wave scheduling modules
        from algorithms.memetic_algorithm import MemeticAlgorithm, create_sample_wave_data
        from models.wave_scheduling_model import WaveSchedulingModel, create_sample_maritime_network
        from utils.oceanographic_utils import OceanographicProcessor, create_indonesian_forecast
        
        st.header("🌊 Wave-Influenced Maritime Scheduling")
        
        # Status indicator
        st.success("✅ All wave scheduling modules loaded successfully!")
        
        tab1, tab2, tab3 = st.tabs(["🌊 Wave Forecast", "⚓ Maritime Scheduling", "🧬 Memetic Optimization"])
        
        with tab1:
            st.subheader("🌊 Oceanographic Conditions")
            
            # Location selector
            locations = ["Jakarta_Bay", "Surabaya_Coast", "Balikpapan_Waters", "Makassar_Strait", "Banda_Sea"]
            selected_location = st.selectbox("Select Maritime Location:", locations)
            
            if st.button("Generate Wave Forecast", type="primary"):
                with st.spinner("Generating oceanographic forecast..."):
                    forecast = create_indonesian_forecast(selected_location)
                    
                    st.success(f"Generated {len(forecast)} hour forecast for {selected_location}")
                    
                    # Display sample conditions
                    if forecast:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Avg Wave Height", f"{sum(c.significant_wave_height for c in forecast)/len(forecast):.1f}m")
                        with col2:
                            st.metric("Max Wave Height", f"{max(c.significant_wave_height for c in forecast):.1f}m")
                        with col3:
                            st.metric("Avg Wind Speed", f"{sum(c.wind_speed for c in forecast)/len(forecast):.1f} m/s")
                        with col4:
                            st.metric("Calm Conditions", f"{sum(1 for c in forecast if c.sea_state.value <= 2)}/{len(forecast)}")
                        
                        # Show first few conditions
                        st.subheader("Sample Conditions")
                        for i, condition in enumerate(forecast[:6]):
                            st.write(f"**{condition.timestamp.strftime('%H:%M')}**: Wave {condition.significant_wave_height:.1f}m, Wind {condition.wind_speed:.1f}m/s, {condition.sea_state.name}")
        
        with tab2:
            st.subheader("⚓ Maritime Network Scheduling")
            
            if st.button("Create Sample Maritime Network", type="secondary"):
                with st.spinner("Setting up maritime network..."):
                    model = create_sample_maritime_network()
                    
                    export_data = model.export_schedule_data()
                    
                    st.success("Maritime network created successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Maritime Locations", len(export_data['maritime_locations']))
                        st.metric("Vessels Available", len(export_data['vessels']))
                    
                    with col2:
                        st.metric("Delivery Tasks", len(export_data['delivery_tasks']))
                        st.metric("Weather Windows", export_data['weather_windows'])
        
        with tab3:
            st.subheader("🧬 Memetic Algorithm Demo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                population_size = st.slider("Population Size", 20, 100, 50)
                generations = st.slider("Generations", 20, 100, 50)
            
            with col2:
                locations = st.multiselect("Coastal Locations", 
                                         ['SPBU_Coast_01', 'SPBU_Coast_02', 'SPBU_Island_01', 'SPBU_Offshore_01'],
                                         default=['SPBU_Coast_01', 'SPBU_Coast_02'])
            
            if st.button("Run Memetic Optimization", type="primary") and locations:
                with st.spinner("Running memetic algorithm optimization..."):
                    from datetime import datetime
                    
                    # Setup
                    demands = {loc: 5000 for loc in locations}
                    wave_data = create_sample_wave_data(datetime.now(), days=3)
                    
                    # Run optimization
                    memetic = MemeticAlgorithm(wave_data, locations, demands, population_size, generations)
                    results = memetic.evolve()
                    
                    st.success("Memetic optimization completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Best Fitness", f"{results['best_fitness']:.3f}")
                    with col2:
                        st.metric("Total Cost", f"IDR {results['total_cost']:,.0f}")
                    with col3:
                        st.metric("Computation Time", f"{results['computation_time']:.2f}s")
                    
                    # Show optimized schedule
                    st.subheader("Optimized Schedule")
                    for delivery in results['best_schedule'][:3]:
                        st.write(f"**{delivery['location']}**: {delivery['delivery_time']} (Wave: {delivery['wave_height']:.1f}m)")
    
    except ImportError as e:
        st.error(f"❌ Wave scheduling modules not found: {e}")
        st.markdown("""
        **Missing Phase 3 Modules:**
        - `algorithms/memetic_algorithm.py`
        - `models/wave_scheduling_model.py`
        - `utils/oceanographic_utils.py`
        """)
        
        # Show Phase 3 preview
        st.markdown("---")
        st.subheader("🌊 Wave Scheduling Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Memetic Algorithm Features:**
            - Wave-influenced scheduling
            - Genetic algorithm + local search
            - Maritime safety constraints
            - Weather window optimization
            - Multi-objective scheduling
            """)
        
        with col2:
            st.markdown("""
            **Oceanographic Integration:**
            - Indonesian waters modeling
            - Tidal analysis and prediction
            - Wave height forecasting
            - Vessel operability assessment
            - Maritime safety scoring
            """)

def render_comparison_page():
    """Enhanced comparison page"""
    st.header("📊 Integrated Comparison Dashboard")
    
    # System integration status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Phase 1 Modules", "2", delta="ARIMA Analysis", help="BBM + Transport Analysis")
    with col2:
        st.metric("Phase 2 Modules", "5", delta="Optimization", help="GA, VRP, Geographic, Export")
    with col3:
        st.metric("Phase 3 Modules", "3", delta="Maritime", help="Memetic, Wave, Oceanographic")
    
    # Integration opportunities
    st.markdown("---")
    st.subheader("🔗 System Integration Opportunities")
    
    integration_scenarios = [
        ("ARIMA → MARLOFIR-P", "Use demand forecasts as optimization input", "✅ Available"),
        ("MARLOFIR-P → Wave Scheduling", "Use optimized routes for maritime scheduling", "✅ Available"),
        ("Complete Pipeline", "ARIMA → GA → Wave scheduling end-to-end", "🔄 In Development"),
        ("Comparative Analysis", "Compare optimization scenarios across all phases", "🔜 Planned")
    ]
    
    for scenario, description, status in integration_scenarios:
        if status.startswith("✅"):
            st.success(f"**{scenario}**: {description} - {status}")
        elif status.startswith("🔄"):
            st.warning(f"**{scenario}**: {description} - {status}")
        else:
            st.info(f"**{scenario}**: {description} - {status}")
    
    # Quick navigation
    st.markdown("---")
    st.subheader("🚀 Quick Navigation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Start with Forecasting", use_container_width=True):
            st.session_state.main_navigation = "⛽ BBM Analysis"
            st.rerun()
    
    with col2:
        if st.button("🧬 Try Optimization", use_container_width=True):
            st.session_state.main_navigation = "🔬 MARLOFIR-P Optimization"
            st.rerun()
    
    with col3:
        if st.button("🌊 Test Wave Scheduling", use_container_width=True):
            st.session_state.main_navigation = "🌊 Wave Scheduling"
            st.rerun()

if __name__ == "__main__":
    main()