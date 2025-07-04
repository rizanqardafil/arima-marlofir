"""
BBM Dashboard - Main UI
Streamlit interface for BBM forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import our analysis functions
from bbm_analysis import (
    run_arima_analysis, 
    create_forecast_chart, 
    generate_bbm_data,
    generate_vehicle_data,
    generate_wave_data,
    create_summary_table,
    create_forecast_table
)

# Page config
st.set_page_config(
    page_title="BBM Dashboard",
    page_icon="⛽",
    layout="wide"
)

# Title
st.title("⛽ Dashboard BBM - Complete Analysis")
st.markdown("**Input Data & Analisis ARIMA dalam 1 Dashboard**")
st.markdown("---")

# Initialize session state
if 'bbm_data' not in st.session_state:
    st.session_state.bbm_data = {}
if 'vehicle_data' not in st.session_state:
    st.session_state.vehicle_data = {}
if 'wave_data' not in st.session_state:
    st.session_state.wave_data = {}
if 'transport_data' not in st.session_state:
    st.session_state.transport_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Transport mode configuration
TRANSPORT_MODES = {
    '🚤 Kapal Nelayan': {'base_consumption': 800, 'efficiency': 0.8},
    '🏍️ Ojek Pangkalan': {'base_consumption': 150, 'efficiency': 1.2},
    '🚗 Mobil Pribadi': {'base_consumption': 200, 'efficiency': 1.0},
    '🚛 Truck Angkutan': {'base_consumption': 500, 'efficiency': 0.7},
    '⛵ Kapal Penumpang': {'base_consumption': 1200, 'efficiency': 0.6},
    '🏭 Generator/Mesin': {'base_consumption': 300, 'efficiency': 0.9}
}

def calculate_bbm_consumption(unit_count, mode_name, base_consumption, days_in_month=30):
    """Calculate BBM consumption for transport mode"""
    mode_data = TRANSPORT_MODES[mode_name]
    daily_consumption = base_consumption * mode_data['efficiency']
    monthly_consumption = unit_count * daily_consumption * days_in_month
    return monthly_consumption

def generate_transport_data(locations, num_months, transport_params, dates):
    """Generate transport mode BBM consumption data"""
    all_data = {}
    
    for loc in locations:
        all_data[loc] = {'dates': dates, 'modes': {}}
        
        for mode_name in TRANSPORT_MODES.keys():
            unit_count = transport_params.get(f"{loc}_{mode_name}_units", 0)
            base_cons = transport_params.get(f"{loc}_{mode_name}_base", TRANSPORT_MODES[mode_name]['base_consumption'])
            variation = transport_params.get(f"{loc}_{mode_name}_var", 15)
            
            if unit_count > 0:  # Only generate data if units exist
                np.random.seed(hash(f"{loc}_{mode_name}") % 1000)
                
                monthly_consumption = []
                for i in range(num_months):
                    seasonal = 1 + 0.15 * np.sin(2 * np.pi * i / 12)
                    trend = 1 + 0.01 * i
                    random_var = np.random.normal(1.0, variation/100)
                    
                    consumption = calculate_bbm_consumption(unit_count, mode_name, base_cons)
                    final_consumption = consumption * seasonal * trend * random_var
                    monthly_consumption.append(max(final_consumption, consumption * 0.5))
                
                all_data[loc]['modes'][mode_name] = monthly_consumption
    
    return all_data

# Sidebar untuk input (Step 1)
with st.sidebar:
    st.header("📋 Step 1: Input Data BBM")
    
    # Step 1: Lokasi
    st.subheader("1. Setup Lokasi")
    num_locations = st.number_input("Berapa lokasi?", 1, 5, 2)
    
    locations = []
    for i in range(num_locations):
        default_name = ["Jemaja", "Siantan", "Palmatak", "Kute Siantan", "Siantan Timur"][i] if i < 5 else f"Lokasi_{i+1}"
        loc_name = st.text_input(f"Nama Lokasi {i+1}:", value=default_name, key=f"loc_{i}")
        locations.append(loc_name)
    
    st.markdown("---")
    
    # Step 2: Periode (moved before vehicle/wave data generation)
    st.subheader("2. Setup Periode")
    num_months = st.slider("Berapa bulan data historis?", 8, 18, 12)
    
    # Generate dates
    start_date = datetime(2023, 1, 31)
    dates = [start_date + timedelta(days=30*i) for i in range(num_months)]
    
    st.markdown("---")
    
    # Step 3: Input Data per Lokasi
    st.subheader("3. Input Data BBM")
    
    # Collect all parameters
    base_params = {}
    
    for loc in locations:
        with st.expander(f"📍 {loc}", expanded=False):
            
            # BBM Tipe 1
            st.write("**BBM Tipe 1 (Liter/bulan):**")
            base_params[f"{loc}_base_t1"] = st.slider(
                f"Base consumption:", 
                1000, 20000, 5000, 
                key=f"{loc}_base_t1"
            )
            
            base_params[f"{loc}_var_t1"] = st.slider(
                f"Variasi (±%):", 
                5, 50, 20, 
                key=f"{loc}_var_t1"
            )
            
            # BBM Tipe 2
            st.write("**BBM Tipe 2 (Liter/bulan):**")
            base_params[f"{loc}_base_t2"] = st.slider(
                f"Base consumption:", 
                2000, 40000, 10000, 
                key=f"{loc}_base_t2"
            )
            
            base_params[f"{loc}_var_t2"] = st.slider(
                f"Variasi (±%):", 
                5, 50, 25, 
                key=f"{loc}_var_t2"
            )
    
    # Generate BBM data
    bbm_data = generate_bbm_data(locations, num_months, base_params)
    
    # Store BBM data in session state with dates
    for loc in locations:
        st.session_state.bbm_data[loc] = {
            'dates': dates[:num_months],
            'bbm_tipe_1': bbm_data[loc]['bbm_tipe_1'],
            'bbm_tipe_2': bbm_data[loc]['bbm_tipe_2']
        }
    
    st.markdown("---")
    
    # Step 4: Vehicle Data
    st.subheader("4. Data Kendaraan (Opsional)")
    
    include_vehicles = st.checkbox("Sertakan Analisis Kendaraan", value=True)
    
    if include_vehicles:
        vehicle_params = {}
        vehicle_types = ['Kendaraan Air', 'Roda Dua', 'Roda Tiga', 'Roda Empat', 'Roda Lima', 'Alat Berat']
        
        with st.expander("🚗 Setup Data Kendaraan", expanded=False):
            for i, vehicle_type in enumerate(vehicle_types):
                st.write(f"**{vehicle_type}:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    default_base = [100, 500, 200, 300, 50, 25][i]
                    vehicle_params[f"vehicle_{i}_base"] = st.number_input(
                        f"Jumlah {vehicle_type}:", 
                        10, 2000, default_base,
                        key=f"vehicle_{i}_base"
                    )
                
                with col2:
                    vehicle_params[f"vehicle_{i}_var"] = st.slider(
                        f"Variasi (±%):", 
                        5, 30, 15,
                        key=f"vehicle_{i}_var"
                    )
        
        # Generate vehicle data
        vehicle_data = generate_vehicle_data(num_months, vehicle_params)
        
        # Store in session state
        for vehicle_type, values in vehicle_data.items():
            st.session_state.vehicle_data[vehicle_type] = {
                'dates': dates[:num_months],
                'values': values
            }
    
    st.markdown("---")
    
    # Step 5: Wave Data
    st.subheader("5. Data Gelombang (Opsional)")
    
    include_waves = st.checkbox("Sertakan Analisis Gelombang", value=True)
    
    if include_waves:
        wave_params = {}
        wave_locations = ['Pantai Utara', 'Pantai Selatan', 'Pantai Timur', 'Pantai Barat']
        
        with st.expander("🌊 Setup Data Gelombang", expanded=False):
            for i, wave_location in enumerate(wave_locations):
                st.write(f"**{wave_location}:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    default_base = [1.5, 2.0, 1.8, 1.6][i]
                    wave_params[f"wave_{i}_base"] = st.number_input(
                        f"Tinggi Gelombang Rata-rata (m):", 
                        0.5, 5.0, default_base,
                        key=f"wave_{i}_base"
                    )
                
                with col2:
                    wave_params[f"wave_{i}_var"] = st.slider(
                        f"Variasi (±%):", 
                        10, 50, 30,
                        key=f"wave_{i}_var"
                    )
        
        # Generate wave data
        wave_data = generate_wave_data(num_months, wave_params)
        
        # Store in session state
        for wave_location, values in wave_data.items():
            st.session_state.wave_data[wave_location] = {
                'dates': dates[:num_months],
                'values': values
            }
    
    st.markdown("---")
    
    # Step 6: Transport Mode Data (NEW)
    st.subheader("6. Data Transport Mode (Opsional)")
    
    include_transport = st.checkbox("Sertakan Analisis Transport Mode", value=False)
    
    if include_transport:
        transport_params = {}
        
        with st.expander("🚗 Setup Transport Mode", expanded=False):
            for loc in locations:
                st.write(f"**🏝️ {loc}**")
                
                for mode_name, mode_config in TRANSPORT_MODES.items():
                    with st.container():
                        st.write(f"*{mode_name}*")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            transport_params[f"{loc}_{mode_name}_units"] = st.number_input(
                                "Jumlah Unit:", 
                                0, 500, 10,
                                key=f"{loc}_{mode_name}_units"
                            )
                        
                        with col2:
                            transport_params[f"{loc}_{mode_name}_base"] = st.number_input(
                                "Konsumsi/Unit (L/hari):", 
                                10, 2000, mode_config['base_consumption'],
                                key=f"{loc}_{mode_name}_base"
                            )
                        
                        with col3:
                            transport_params[f"{loc}_{mode_name}_var"] = st.slider(
                                "Variasi (%):", 
                                5, 40, 15,
                                key=f"{loc}_{mode_name}_var"
                            )
                
                st.markdown("---")
        
        # Generate transport data
        transport_data = generate_transport_data(locations, num_months, transport_params, dates)
        st.session_state.transport_data = transport_data
    
    st.markdown("---")
    
    # Step 7: ARIMA Analysis
    st.header("🔬 Step 2: ARIMA Analysis")
    
    forecast_months = st.slider("Forecast periode (bulan):", 6, 24, 12)
    
    if st.button("🚀 Run ARIMA Analysis", type="primary"):
        if st.session_state.bbm_data:
            st.session_state.run_analysis = True
            st.session_state.forecast_months = forecast_months
        else:
            st.error("⚠️ Input data BBM dulu!")


# Main content area
if st.session_state.get('run_analysis', False) and st.session_state.bbm_data:
    
    st.header("🔬 ARIMA Analysis Results")
    
    with st.spinner("🔄 Running ARIMA analysis... Please wait..."):
        
        # Run analysis for all data types
        results = {}
        progress_bar = st.progress(0)
        
        # Count total analyses
        total_analyses = len(st.session_state.bbm_data) * 2  # BBM
        if st.session_state.vehicle_data:
            total_analyses += len(st.session_state.vehicle_data)  # Vehicles
        if st.session_state.wave_data:
            total_analyses += len(st.session_state.wave_data)  # Waves
        if st.session_state.transport_data:
            for loc_data in st.session_state.transport_data.values():
                total_analyses += len([mode for mode, data in loc_data['modes'].items() if len(data) > 0])  # Transport modes
        
        current_analysis = 0
        
        # BBM Analysis
        for loc, data in st.session_state.bbm_data.items():
            for bbm_type, values in [('BBM Tipe 1', data['bbm_tipe_1']), 
                                     ('BBM Tipe 2', data['bbm_tipe_2'])]:
                
                series = pd.Series(values)
                result = run_arima_analysis(
                    series, loc, bbm_type, 
                    st.session_state.forecast_months
                )
                results[(loc, bbm_type)] = result
                
                current_analysis += 1
                progress_bar.progress(current_analysis / total_analyses)
        
        # Vehicle Analysis
        for vehicle_type, data in st.session_state.vehicle_data.items():
            series = pd.Series(data['values'])
            result = run_arima_analysis(
                series, 'All_Regions', vehicle_type,
                st.session_state.forecast_months
            )
            results[('All_Regions', vehicle_type)] = result
            
            current_analysis += 1
            progress_bar.progress(current_analysis / total_analyses)
        
        # Wave Analysis  
        for wave_location, data in st.session_state.wave_data.items():
            series = pd.Series(data['values'])
            result = run_arima_analysis(
                series, wave_location, 'Tinggi_Gelombang',
                st.session_state.forecast_months
            )
            results[(wave_location, 'Tinggi_Gelombang')] = result
            
            current_analysis += 1
            progress_bar.progress(current_analysis / total_analyses)
        
        # Transport Mode Analysis (NEW)
        for loc, loc_data in st.session_state.transport_data.items():
            for mode_name, consumption_data in loc_data['modes'].items():
                if len(consumption_data) > 0 and sum(consumption_data) > 0:
                    series = pd.Series(consumption_data)
                    result = run_arima_analysis(
                        series, loc, f"Transport_{mode_name}",
                        st.session_state.forecast_months
                    )
                    results[(loc, f"Transport_{mode_name}")] = result
                    
                    current_analysis += 1
                    progress_bar.progress(current_analysis / total_analyses)
        
        st.session_state.analysis_results = results
        progress_bar.empty()
    
    st.success("✅ ARIMA Analysis Complete!")
    
    # Display results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast Charts", "🚗 Transport Analysis", "📊 Model Summary", "📋 Detailed Results"])
    
    with tab1:
        st.subheader("📈 Forecast Visualizations")
        
        for (loc, data_type), result in st.session_state.analysis_results.items():
            if 'error' not in result:
                # Determine data source and type
                if data_type in ['BBM Tipe 1', 'BBM Tipe 2']:
                    historical_data = st.session_state.bbm_data[loc][data_type.lower().replace(' ', '_')]
                    dates = st.session_state.bbm_data[loc]['dates']
                elif data_type.startswith('Kendaraan') or data_type in ['Kendaraan Air', 'Roda Dua', 'Roda Tiga', 'Roda Empat', 'Roda Lima', 'Alat Berat']:
                    historical_data = st.session_state.vehicle_data[data_type]['values']
                    dates = st.session_state.vehicle_data[data_type]['dates']
                elif data_type == 'Tinggi_Gelombang':
                    historical_data = st.session_state.wave_data[loc]['values']
                    dates = st.session_state.wave_data[loc]['dates']
                elif data_type.startswith('Transport_'):
                    mode_name = data_type.replace('Transport_', '')
                    historical_data = st.session_state.transport_data[loc]['modes'][mode_name]
                    dates = st.session_state.transport_data[loc]['dates']
                else:
                    continue
                
                # Create chart
                fig = create_forecast_chart(historical_data, result, dates)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("ARIMA Order", f"{result['arima_order']}")
                    with col2:
                        st.metric("AIC", f"{result['aic']:.2f}")
                    with col3:
                        st.metric("RMSE", f"{result['rmse']:.2f}" if result['rmse'] else "N/A")
                    with col4:
                        mape_val = f"{result['mape']:.1f}%" if result['mape'] else "N/A"
                        st.metric("MAPE", mape_val)
            else:
                st.error(f"❌ Error in {loc} - {data_type}: {result['error']}")
    
    with tab2:
        st.subheader("🚗 Transport Mode Analysis")
        
        if st.session_state.transport_data:
            # Transport mode breakdown charts
            for loc in locations:
                if loc in st.session_state.transport_data:
                    loc_data = st.session_state.transport_data[loc]
                    
                    if any(len(data) > 0 for data in loc_data['modes'].values()):
                        st.write(f"### 🏝️ {loc}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart for total consumption
                            mode_totals = {}
                            for mode_name, consumption_data in loc_data['modes'].items():
                                total = sum(consumption_data) if len(consumption_data) > 0 else 0
                                if total > 0:
                                    mode_totals[mode_name] = total
                            
                            if mode_totals:
                                fig_pie = px.pie(
                                    values=list(mode_totals.values()),
                                    names=list(mode_totals.keys()),
                                    title=f"BBM Distribution by Transport Mode - {loc}"
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            # Line chart for monthly trends
                            mode_data_for_chart = []
                            for mode_name, consumption_data in loc_data['modes'].items():
                                if len(consumption_data) > 0 and sum(consumption_data) > 0:
                                    for i, (date, consumption) in enumerate(zip(loc_data['dates'], consumption_data)):
                                        mode_data_for_chart.append({
                                            'Date': date,
                                            'Mode': mode_name,
                                            'Consumption': consumption
                                        })
                            
                            if mode_data_for_chart:
                                df_chart = pd.DataFrame(mode_data_for_chart)
                                fig_line = px.line(
                                    df_chart, 
                                    x='Date', 
                                    y='Consumption', 
                                    color='Mode',
                                    title=f"Monthly Consumption Trends - {loc}"
                                )
                                st.plotly_chart(fig_line, use_container_width=True)
                        
                        # Summary table
                        stats_data = []
                        for mode_name, consumption_data in loc_data['modes'].items():
                            if len(consumption_data) > 0:
                                total = sum(consumption_data)
                                if total > 0:
                                    avg_monthly = np.mean(consumption_data)
                                    percentage = (total / sum([sum(data) for data in loc_data['modes'].values() if len(data) > 0])) * 100
                                    
                                    # Get forecast if available
                                    forecast_key = (loc, f"Transport_{mode_name}")
                                    forecast_total = "N/A"
                                    if forecast_key in st.session_state.analysis_results and 'forecast' in st.session_state.analysis_results[forecast_key]:
                                        forecast_total = f"{sum(st.session_state.analysis_results[forecast_key]['forecast']):,.0f} L"
                                    
                                    stats_data.append({
                                        'Transport Mode': mode_name,
                                        'Total Historical': f"{total:,.0f} L",
                                        'Avg Monthly': f"{avg_monthly:,.0f} L",
                                        'Percentage': f"{percentage:.1f}%",
                                        'Forecast Total': forecast_total
                                    })
                        
                        if stats_data:
                            df_stats = pd.DataFrame(stats_data)
                            st.dataframe(df_stats, use_container_width=True)
                        
                        st.markdown("---")
        else:
            st.info("No transport mode data available. Enable transport mode analysis in the sidebar.")
    
    with tab3:
        st.subheader("📊 Model Performance Summary")
        
        # Create summary table menggunakan fungsi dari bbm_analysis.py
        summary_df = create_summary_table(st.session_state.analysis_results)
        
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                "📥 Download Summary",
                csv,
                "arima_analysis_summary.csv",
                "text/csv"
            )
    
    with tab4:
        st.subheader("📋 Detailed Forecast Results")
        
        selected_location = st.selectbox(
            "Pilih lokasi/kategori untuk detail forecast:",
            list(set([loc for loc, _ in st.session_state.analysis_results.keys()]))
        )
        
        # Show available data types for selected location
        available_types = [data_type for loc, data_type in st.session_state.analysis_results.keys() if loc == selected_location]
        
        for data_type in available_types:
            if (selected_location, data_type) in st.session_state.analysis_results:
                result = st.session_state.analysis_results[(selected_location, data_type)]
                
                if 'error' not in result:
                    st.write(f"**{data_type}:**")
                    
                    # Get dates based on data type
                    if data_type in ['BBM Tipe 1', 'BBM Tipe 2']:
                        dates = st.session_state.bbm_data[selected_location]['dates']
                    elif data_type.startswith('Kendaraan') or data_type in ['Kendaraan Air', 'Roda Dua', 'Roda Tiga', 'Roda Empat', 'Roda Lima', 'Alat Berat']:
                        dates = st.session_state.vehicle_data[data_type]['dates']
                    elif data_type == 'Tinggi_Gelombang':
                        dates = st.session_state.wave_data[selected_location]['dates']
                    elif data_type.startswith('Transport_'):
                        dates = st.session_state.transport_data[selected_location]['dates']
                    else:
                        continue
                    
                    # Create forecast table
                    forecast_df = create_forecast_table(result, dates)
                    
                    if not forecast_df.empty:
                        st.dataframe(forecast_df)
                    st.markdown("---")

elif st.session_state.bbm_data or st.session_state.vehicle_data or st.session_state.wave_data or st.session_state.transport_data:
    # Show Step 1 results
    st.header("📊 Step 1: Data Preview")
    
    # Create tabs for different data types
    available_tabs = []
    if st.session_state.bbm_data:
        available_tabs.append("⛽ BBM Data")
    if st.session_state.vehicle_data:
        available_tabs.append("🚗 Vehicle Data")  
    if st.session_state.wave_data:
        available_tabs.append("🌊 Wave Data")
    if st.session_state.transport_data:
        available_tabs.append("🚛 Transport Mode")
    
    if len(available_tabs) > 1:
        tabs = st.tabs(available_tabs)
        tab_idx = 0
        
        # BBM Data Tab
        if st.session_state.bbm_data:
            with tabs[tab_idx]:
                tab_idx += 1
                st.subheader("📋 BBM Data Table")
                
                # Create two columns for table and stats
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    preview_data = []
                    for loc, data in st.session_state.bbm_data.items():
                        for i, date in enumerate(data['dates']):
                            preview_data.append({
                                'Tanggal': date.strftime('%b %Y'),
                                'Lokasi': loc,
                                'BBM Tipe 1': f"{data['bbm_tipe_1'][i]:,.0f} L",
                                'BBM Tipe 2': f"{data['bbm_tipe_2'][i]:,.0f} L"
                            })
                    
                    preview_df = pd.DataFrame(preview_data)
                    st.dataframe(preview_df, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Quick Stats")
                    for loc, data in st.session_state.bbm_data.items():
                        st.write(f"**{loc}:**")
                        st.write(f"• T1 Avg: {np.mean(data['bbm_tipe_1']):,.0f} L")
                        st.write(f"• T2 Avg: {np.mean(data['bbm_tipe_2']):,.0f} L")
                        st.write(f"• Periode: {len(data['dates'])} bulan")
                        st.write("---")
                
                # Quick visualization
                plot_data = []
                for loc, data in st.session_state.bbm_data.items():
                    for i, date in enumerate(data['dates']):
                        plot_data.append({
                            'Tanggal': date,
                            'Lokasi': loc,
                            'BBM Tipe 1': data['bbm_tipe_1'][i],
                            'BBM Tipe 2': data['bbm_tipe_2'][i]
                        })
                
                plot_df = pd.DataFrame(plot_data)
                fig = px.line(plot_df, x='Tanggal', y=['BBM Tipe 1', 'BBM Tipe 2'], 
                             color='Lokasi', title="Preview BBM Data")
                st.plotly_chart(fig, use_container_width=True)
        
        # Vehicle Data Tab
        if st.session_state.vehicle_data:
            with tabs[tab_idx]:
                tab_idx += 1
                st.subheader("📋 Vehicle Data Table")
                
                vehicle_preview = []
                for vehicle_type, data in st.session_state.vehicle_data.items():
                    for i, date in enumerate(data['dates']):
                        vehicle_preview.append({
                            'Tanggal': date.strftime('%b %Y'),
                            'Jenis Kendaraan': vehicle_type,
                            'Jumlah': f"{data['values'][i]:,} unit"
                        })
                
                vehicle_df = pd.DataFrame(vehicle_preview)
                st.dataframe(vehicle_df, use_container_width=True)
                
                # Vehicle chart
                vehicle_plot_data = []
                for vehicle_type, data in st.session_state.vehicle_data.items():
                    for i, date in enumerate(data['dates']):
                        vehicle_plot_data.append({
                            'Tanggal': date,
                            'Jenis Kendaraan': vehicle_type,
                            'Jumlah': data['values'][i]
                        })
                
                vehicle_plot_df = pd.DataFrame(vehicle_plot_data)
                fig_vehicle = px.line(vehicle_plot_df, x='Tanggal', y='Jumlah', 
                                    color='Jenis Kendaraan', title="Preview Vehicle Data")
                st.plotly_chart(fig_vehicle, use_container_width=True)
        
        # Wave Data Tab
        if st.session_state.wave_data:
            with tabs[tab_idx]:
                tab_idx += 1
                st.subheader("📋 Wave Data Table")
                
                wave_preview = []
                for wave_location, data in st.session_state.wave_data.items():
                    for i, date in enumerate(data['dates']):
                        wave_preview.append({
                            'Tanggal': date.strftime('%b %Y'),
                            'Lokasi': wave_location,
                            'Tinggi Gelombang': f"{data['values'][i]:.2f} m"
                        })
                
                wave_df = pd.DataFrame(wave_preview)
                st.dataframe(wave_df, use_container_width=True)
                
                # Wave chart
                wave_plot_data = []
                for wave_location, data in st.session_state.wave_data.items():
                    for i, date in enumerate(data['dates']):
                        wave_plot_data.append({
                            'Tanggal': date,
                            'Lokasi': wave_location,
                            'Tinggi Gelombang': data['values'][i]
                        })
                
                wave_plot_df = pd.DataFrame(wave_plot_data)
                fig_wave = px.line(wave_plot_df, x='Tanggal', y='Tinggi Gelombang', 
                                 color='Lokasi', title="Preview Wave Data")
                st.plotly_chart(fig_wave, use_container_width=True)
        
        # Transport Mode Data Tab (NEW)
        if st.session_state.transport_data:
            with tabs[tab_idx]:
                st.subheader("📋 Transport Mode Data Table")
                
                transport_preview = []
                for loc, loc_data in st.session_state.transport_data.items():
                    for mode_name, consumption_data in loc_data['modes'].items():
                        if len(consumption_data) > 0 and sum(consumption_data) > 0:
                            for i, date in enumerate(loc_data['dates']):
                                transport_preview.append({
                                    'Tanggal': date.strftime('%b %Y'),
                                    'Lokasi': loc,
                                    'Transport Mode': mode_name,
                                    'Konsumsi BBM': f"{consumption_data[i]:,.0f} L"
                                })
                
                if transport_preview:
                    transport_df = pd.DataFrame(transport_preview)
                    st.dataframe(transport_df, use_container_width=True)
                    
                    # Transport mode charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Total consumption by mode across all locations
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
                            fig_pie_transport = px.pie(
                                values=list(mode_totals.values()),
                                names=list(mode_totals.keys()),
                                title="Total BBM by Transport Mode"
                            )
                            st.plotly_chart(fig_pie_transport, use_container_width=True)
                    
                    with col2:
                        # Monthly trends
                        if transport_preview:
                            transport_plot_df = pd.DataFrame(transport_preview)
                            transport_plot_df['Konsumsi BBM (Numeric)'] = transport_plot_df['Konsumsi BBM'].str.replace(' L', '').str.replace(',', '').astype(float)
                            transport_plot_df['Tanggal'] = pd.to_datetime(transport_plot_df['Tanggal'], format='%b %Y')
                            
                            fig_transport_line = px.line(
                                transport_plot_df, 
                                x='Tanggal', 
                                y='Konsumsi BBM (Numeric)', 
                                color='Transport Mode',
                                title="Monthly Transport Mode Trends"
                            )
                            st.plotly_chart(fig_transport_line, use_container_width=True)
                else:
                    st.info("No active transport modes with consumption data.")
    
    else:
        # Single tab - BBM data only
        if st.session_state.bbm_data:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("📋 BBM Data Table")
                preview_data = []
                for loc, data in st.session_state.bbm_data.items():
                    for i, date in enumerate(data['dates']):
                        preview_data.append({
                            'Tanggal': date.strftime('%b %Y'),
                            'Lokasi': loc,
                            'BBM Tipe 1': f"{data['bbm_tipe_1'][i]:,.0f} L",
                            'BBM Tipe 2': f"{data['bbm_tipe_2'][i]:,.0f} L"
                        })
                
                preview_df = pd.DataFrame(preview_data)
                st.dataframe(preview_df, use_container_width=True)
            
            with col2:
                st.subheader("📊 Quick Stats")
                for loc, data in st.session_state.bbm_data.items():
                    st.write(f"**{loc}:**")
                    st.write(f"• T1 Avg: {np.mean(data['bbm_tipe_1']):,.0f} L")
                    st.write(f"• T2 Avg: {np.mean(data['bbm_tipe_2']):,.0f} L")
                    st.write(f"• Periode: {len(data['dates'])} bulan")
                    st.write("---")
            
            # Chart
            plot_data = []
            for loc, data in st.session_state.bbm_data.items():
                for i, date in enumerate(data['dates']):
                    plot_data.append({
                        'Tanggal': date,
                        'Lokasi': loc,
                        'BBM Tipe 1': data['bbm_tipe_1'][i],
                        'BBM Tipe 2': data['bbm_tipe_2'][i]
                    })
            
            plot_df = pd.DataFrame(plot_data)
            fig = px.line(plot_df, x='Tanggal', y=['BBM Tipe 1', 'BBM Tipe 2'], 
                         color='Lokasi', title="Preview BBM Data")
            st.plotly_chart(fig, use_container_width=True)
    
    st.info("👈 Klik 'Run ARIMA Analysis' di sidebar untuk melanjutkan ke Step 2")

else:
    st.info("👈 Silakan input data BBM di sidebar untuk memulai")
    
    st.subheader("📖 Cara Penggunaan:")
    st.markdown("""
    **Step 1 - Input Data:**
    1. Setup lokasi dan periode (min 8 bulan untuk ARIMA)
    2. Atur base consumption dan variasi untuk BBM
    3. Opsional: Aktifkan analisis Kendaraan dan Gelombang
    4. **BARU**: Aktifkan analisis Transport Mode untuk analisis per moda transportasi
    5. Preview semua data yang sudah diinput
    
    **Step 2 - ARIMA Analysis:**
    6. Pilih periode forecast (6-24 bulan)
    7. Klik 'Run ARIMA Analysis' 
    8. Lihat hasil forecast untuk BBM, Kendaraan, Gelombang, dan Transport Mode
    9. Download summary dan detailed results
    
    **Data Types Available:**
    - ⛽ **BBM**: Tipe 1 & 2 per lokasi
    - 🚗 **Kendaraan**: 6 jenis (Air, Roda 2-5, Alat Berat)
    - 🌊 **Gelombang**: 4 lokasi pantai
    - 🚛 **Transport Mode**: 6 moda transportasi per lokasi
    """)
    
    # Show transport modes available
    st.subheader("🚛 Available Transport Modes:")
    
    mode_info = []
    for mode_name, config in TRANSPORT_MODES.items():
        mode_info.append({
            'Transport Mode': mode_name,
            'Base Consumption': f"{config['base_consumption']} L/day/unit",
            'Efficiency Factor': f"{config['efficiency']}x"
        })
    
    df_modes = pd.DataFrame(mode_info)
    st.dataframe(df_modes, use_container_width=True)

# Footer
st.markdown("---")
if st.session_state.get('analysis_results'):
    total_models = len(st.session_state.analysis_results)
    bbm_models = len([k for k in st.session_state.analysis_results.keys() if k[1] in ['BBM Tipe 1', 'BBM Tipe 2']])
    vehicle_models = len([k for k in st.session_state.analysis_results.keys() if 'Kendaraan' in k[1] or k[1] in ['Kendaraan Air', 'Roda Dua', 'Roda Tiga', 'Roda Empat', 'Roda Lima', 'Alat Berat']])
    wave_models = len([k for k in st.session_state.analysis_results.keys() if k[1] == 'Tinggi_Gelombang'])
    transport_models = len([k for k in st.session_state.analysis_results.keys() if k[1].startswith('Transport_')])
    
    st.markdown(f"**✅ Analysis Complete!** | Total: {total_models} models (⛽{bbm_models} + 🚗{vehicle_models} + 🌊{wave_models} + 🚛{transport_models})")
else:
    st.markdown("**⏳ Ready for Analysis** | Input data untuk mulai")