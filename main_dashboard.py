"""
BBM Dashboard - Main UI
Streamlit interface for BBM forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Import our analysis functions
from bbm_analysis import (
    run_arima_analysis, 
    create_forecast_chart, 
    generate_bbm_data,
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
st.markdown("---")

# Initialize session state
if 'bbm_data' not in st.session_state:
    st.session_state.bbm_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

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
    
    # Step 2: Periode
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
    
    # Generate data menggunakan fungsi dari bbm_analysis.py
    bbm_data = generate_bbm_data(locations, num_months, base_params)
    
    # Store in session state with dates
    for loc in locations:
        st.session_state.bbm_data[loc] = {
            'dates': dates[:num_months],
            'bbm_tipe_1': bbm_data[loc]['bbm_tipe_1'],
            'bbm_tipe_2': bbm_data[loc]['bbm_tipe_2']
        }
    
    st.markdown("---")
    
    # Step 4: ARIMA Analysis
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
        
        # Run analysis for all locations and BBM types
        results = {}
        progress_bar = st.progress(0)
        total_analyses = len(st.session_state.bbm_data) * 2  # 2 BBM types per location
        current_analysis = 0
        
        for loc, data in st.session_state.bbm_data.items():
            for bbm_type, values in [('BBM Tipe 1', data['bbm_tipe_1']), 
                                     ('BBM Tipe 2', data['bbm_tipe_2'])]:
                
                # Run ARIMA analysis menggunakan fungsi dari bbm_analysis.py
                series = pd.Series(values)
                result = run_arima_analysis(
                    series, loc, bbm_type, 
                    st.session_state.forecast_months
                )
                results[(loc, bbm_type)] = result
                
                current_analysis += 1
                progress_bar.progress(current_analysis / total_analyses)
        
        st.session_state.analysis_results = results
        progress_bar.empty()
    
    st.success("✅ ARIMA Analysis Complete!")
    
    # Display results in tabs
    tab1, tab2, tab3 = st.tabs(["📈 Forecast Charts", "📊 Model Summary", "📋 Detailed Results"])
    
    with tab1:
        st.subheader("📈 Forecast Visualizations")
        
        for (loc, bbm_type), result in st.session_state.analysis_results.items():
            if 'error' not in result:
                historical_data = st.session_state.bbm_data[loc][bbm_type.lower().replace(' ', '_')]
                dates = st.session_state.bbm_data[loc]['dates']
                
                # Create chart menggunakan fungsi dari bbm_analysis.py
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
                        st.metric("RMSE", f"{result['rmse']:.0f}" if result['rmse'] else "N/A")
                    with col4:
                        mape_val = f"{result['mape']:.1f}%" if result['mape'] else "N/A"
                        st.metric("MAPE", mape_val)
            else:
                st.error(f"❌ Error in {loc} - {bbm_type}: {result['error']}")
    
    with tab2:
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
    
    with tab3:
        st.subheader("📋 Detailed Forecast Results")
        
        selected_location = st.selectbox(
            "Pilih lokasi untuk detail forecast:",
            list(set([loc for loc, _ in st.session_state.analysis_results.keys()]))
        )
        
        for bbm_type in ['BBM Tipe 1', 'BBM Tipe 2']:
            if (selected_location, bbm_type) in st.session_state.analysis_results:
                result = st.session_state.analysis_results[(selected_location, bbm_type)]
                
                if 'error' not in result:
                    st.write(f"**{bbm_type}:**")
                    
                    # Create forecast table menggunakan fungsi dari bbm_analysis.py
                    dates = st.session_state.bbm_data[selected_location]['dates']
                    forecast_df = create_forecast_table(result, dates)
                    
                    if not forecast_df.empty:
                        st.dataframe(forecast_df)
                    st.markdown("---")

elif st.session_state.bbm_data:
    # Show Step 1 results
    st.header("📊 Step 1: Data Preview")
    
    # Create preview table
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
    
    # Show preview dalam 2 kolom
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Data Table")
        st.dataframe(preview_df, use_container_width=True)
        
        # Create quick visualization
        st.subheader("📈 Quick Preview")
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
        
        # Simple line chart
        fig = px.line(
            plot_df, 
            x='Tanggal', 
            y=['BBM Tipe 1', 'BBM Tipe 2'], 
            color='Lokasi',
            title="Preview Data BBM"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Quick Stats")
        for loc, data in st.session_state.bbm_data.items():
            st.write(f"**{loc}:**")
            st.write(f"• T1 Avg: {np.mean(data['bbm_tipe_1']):,.0f} L")
            st.write(f"• T2 Avg: {np.mean(data['bbm_tipe_2']):,.0f} L")
            st.write(f"• Periode: {len(data['dates'])} bulan")
            st.write("---")
    
    st.info("👈 Klik 'Run ARIMA Analysis' Untuk Melakukan Arima Analysisgit init")

else:
    st.info("👈 Silakan input data BBM di sidebar untuk memulai")
    
    # Show instruction
    st.subheader("📖 Cara Penggunaan:")
    st.markdown("""
    **Step 1 - Input Data:**
    1. Setup lokasi dan periode (min 8 bulan untuk ARIMA)
    2. Atur base consumption dan variasi untuk setiap lokasi
    3. Preview data yang sudah diinput
    
    **Step 2 - ARIMA Analysis:**
    4. Pilih periode forecast (6-24 bulan)
    5. Klik 'Run ARIMA Analysis' 
    6. Lihat hasil forecast, metrics, dan download summary
    """)

# Footer
st.markdown("---")
if st.session_state.get('analysis_results'):
    st.markdown("**✅ Analysis Complete!** | Dashboard Ready")
else:
    st.markdown("**⏳ Ready for Analysis** | Input data untuk mulai")