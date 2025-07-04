import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="BBM Transport Mode Analysis",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 BBM Analysis by Transport Mode")
st.markdown("**Klasifikasi BBM berdasarkan Moda Transportasi**")
st.markdown("---")

if 'transport_data' not in st.session_state:
    st.session_state.transport_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

TRANSPORT_MODES = {
    '🚤 Kapal Nelayan': {'base_consumption': 800, 'efficiency': 0.8},
    '🏍️ Ojek Pangkalan': {'base_consumption': 150, 'efficiency': 1.2},
    '🚗 Mobil Pribadi': {'base_consumption': 200, 'efficiency': 1.0},
    '🚛 Truck Angkutan': {'base_consumption': 500, 'efficiency': 0.7},
    '⛵ Kapal Penumpang': {'base_consumption': 1200, 'efficiency': 0.6},
    '🏭 Generator/Mesin': {'base_consumption': 300, 'efficiency': 0.9}
}

def calculate_bbm_consumption(unit_count, mode_name, base_consumption, days_in_month=30):
    mode_data = TRANSPORT_MODES[mode_name]
    daily_consumption = base_consumption * mode_data['efficiency']
    monthly_consumption = unit_count * daily_consumption * days_in_month
    return monthly_consumption

def generate_transport_data(locations, num_months, transport_params):
    start_date = datetime(2023, 1, 31)
    dates = [start_date + timedelta(days=30*i) for i in range(num_months)]
    
    all_data = {}
    
    for loc in locations:
        all_data[loc] = {'dates': dates, 'modes': {}}
        
        for mode_name in TRANSPORT_MODES.keys():
            unit_count = transport_params.get(f"{loc}_{mode_name}_units", 0)
            base_cons = transport_params.get(f"{loc}_{mode_name}_base", TRANSPORT_MODES[mode_name]['base_consumption'])
            variation = transport_params.get(f"{loc}_{mode_name}_var", 15)
            
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

def run_arima_forecast(series, forecast_periods=12):
    best_aic = float('inf')
    best_order = (1, 1, 1)
    
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    try:
        model = ARIMA(series, order=best_order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=forecast_periods)
        conf_int = fitted_model.get_forecast(steps=forecast_periods).conf_int()
        
        if len(series) >= 8:
            split_point = int(len(series) * 0.8)
            train_data = series[:split_point]
            test_data = series[split_point:]
            train_model = ARIMA(train_data, order=best_order).fit()
            test_forecast = train_model.forecast(steps=len(test_data))
            rmse = np.sqrt(mean_squared_error(test_data, test_forecast))
            mape = np.mean(np.abs((test_data - test_forecast) / test_data)) * 100
        else:
            rmse = mape = None
        
        return {
            'forecast': forecast,
            'conf_int': conf_int,
            'order': best_order,
            'aic': best_aic,
            'rmse': rmse,
            'mape': mape
        }
    except:
        return None

with st.sidebar:
    st.header("📋 Setup Transport Mode Analysis")
    
    st.subheader("1. Lokasi/Pulau")
    num_locations = st.number_input("Jumlah Lokasi:", 1, 5, 2)
    
    locations = []
    for i in range(num_locations):
        default_name = ["Jemaja", "Siantan", "Palmatak", "Bintan", "Batam"][i] if i < 5 else f"Pulau_{i+1}"
        loc_name = st.text_input(f"Nama Lokasi {i+1}:", value=default_name, key=f"loc_{i}")
        locations.append(loc_name)
    
    st.subheader("2. Periode Data")
    num_months = st.slider("Data historis (bulan):", 8, 18, 12)
    forecast_months = st.slider("Forecast periode (bulan):", 6, 24, 12)
    
    st.subheader("3. Transport Mode Setup")
    
    transport_params = {}
    
    for loc in locations:
        with st.expander(f"🏝️ {loc}", expanded=False):
            
            for mode_name, mode_config in TRANSPORT_MODES.items():
                st.write(f"**{mode_name}**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    transport_params[f"{loc}_{mode_name}_units"] = st.number_input(
                        "Jumlah Unit:", 
                        0, 1000, 10,
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
    
    transport_data = generate_transport_data(locations, num_months, transport_params)
    st.session_state.transport_data = transport_data
    
    st.markdown("---")
    
    if st.button("🚀 Run Analysis", type="primary"):
        if st.session_state.transport_data:
            st.session_state.run_analysis = True
            st.session_state.forecast_months = forecast_months

if st.session_state.get('run_analysis', False):
    
    st.header("📊 Analysis Results")
    
    with st.spinner("🔄 Running ARIMA analysis..."):
        
        results = {}
        progress_bar = st.progress(0)
        
        total_analyses = 0
        for loc_data in st.session_state.transport_data.values():
            total_analyses += len(loc_data['modes'])
        
        current_analysis = 0
        
        for loc, loc_data in st.session_state.transport_data.items():
            for mode_name, consumption_data in loc_data['modes'].items():
                if sum(consumption_data) > 0:
                    series = pd.Series(consumption_data)
                    result = run_arima_forecast(series, st.session_state.forecast_months)
                    
                    if result:
                        results[(loc, mode_name)] = result
                
                current_analysis += 1
                progress_bar.progress(current_analysis / total_analyses)
        
        st.session_state.analysis_results = results
        progress_bar.empty()
    
    st.success("✅ Analysis Complete!")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 BBM Breakdown", "📈 Forecasts", "🏆 Rankings", "📋 Summary"])
    
    with tab1:
        st.subheader("BBM Consumption Breakdown by Transport Mode")
        
        for loc in locations:
            if loc in st.session_state.transport_data:
                loc_data = st.session_state.transport_data[loc]
                
                st.write(f"### 🏝️ {loc}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    mode_totals = {}
                    for mode_name, consumption_data in loc_data['modes'].items():
                        total = sum(consumption_data)
                        if total > 0:
                            mode_totals[mode_name] = total
                    
                    if mode_totals:
                        fig_pie = px.pie(
                            values=list(mode_totals.values()),
                            names=list(mode_totals.keys()),
                            title=f"BBM Distribution - {loc}"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    mode_data_for_chart = []
                    for mode_name, consumption_data in loc_data['modes'].items():
                        if sum(consumption_data) > 0:
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
                            title=f"Monthly Consumption - {loc}"
                        )
                        st.plotly_chart(fig_line, use_container_width=True)
                
                stats_data = []
                for mode_name, consumption_data in loc_data['modes'].items():
                    total = sum(consumption_data)
                    if total > 0:
                        avg_monthly = np.mean(consumption_data)
                        percentage = (total / sum([sum(data) for data in loc_data['modes'].values()])) * 100
                        
                        stats_data.append({
                            'Transport Mode': mode_name,
                            'Total Consumption': f"{total:,.0f} L",
                            'Avg Monthly': f"{avg_monthly:,.0f} L",
                            'Percentage': f"{percentage:.1f}%"
                        })
                
                if stats_data:
                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(df_stats, use_container_width=True)
                
                st.markdown("---")
    
    with tab2:
        st.subheader("Forecast by Transport Mode")
        
        for (loc, mode_name), result in st.session_state.analysis_results.items():
            st.write(f"**{loc} - {mode_name}**")
            
            historical_data = st.session_state.transport_data[loc]['modes'][mode_name]
            dates = st.session_state.transport_data[loc]['dates']
            
            last_date = dates[-1]
            forecast_dates = [last_date + timedelta(days=30*(i+1)) for i in range(len(result['forecast']))]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=historical_data,
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=result['forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            if result['conf_int'] is not None:
                fig.add_trace(go.Scatter(
                    x=forecast_dates + forecast_dates[::-1],
                    y=list(result['conf_int'].iloc[:, 1]) + list(result['conf_int'].iloc[:, 0])[::-1],
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
            
            fig.update_layout(
                title=f"Forecast - {loc} ({mode_name})",
                xaxis_title="Date",
                yaxis_title="BBM Consumption (Liters)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ARIMA Order", f"{result['order']}")
            with col2:
                st.metric("AIC", f"{result['aic']:.2f}")
            with col3:
                st.metric("RMSE", f"{result['rmse']:.0f}" if result['rmse'] else "N/A")
            with col4:
                st.metric("MAPE", f"{result['mape']:.1f}%" if result['mape'] else "N/A")
            
            st.markdown("---")
    
    with tab3:
        st.subheader("Transport Mode Rankings")
        
        all_rankings = []
        
        for loc in locations:
            if loc in st.session_state.transport_data:
                loc_data = st.session_state.transport_data[loc]
                
                for mode_name, consumption_data in loc_data['modes'].items():
                    total = sum(consumption_data)
                    if total > 0:
                        avg_monthly = np.mean(consumption_data)
                        
                        forecast_total = 0
                        if (loc, mode_name) in st.session_state.analysis_results:
                            forecast_result = st.session_state.analysis_results[(loc, mode_name)]
                            forecast_total = sum(forecast_result['forecast'])
                        
                        all_rankings.append({
                            'Location': loc,
                            'Transport Mode': mode_name,
                            'Historical Total': total,
                            'Avg Monthly': avg_monthly,
                            'Forecast Total': forecast_total,
                            'Growth Rate': ((forecast_total / st.session_state.forecast_months) / avg_monthly - 1) * 100 if avg_monthly > 0 and forecast_total > 0 else 0
                        })
        
        if all_rankings:
            df_rankings = pd.DataFrame(all_rankings)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top BBM Consumers (Historical)**")
                top_historical = df_rankings.nlargest(10, 'Historical Total')[['Location', 'Transport Mode', 'Historical Total', 'Avg Monthly']]
                top_historical['Historical Total'] = top_historical['Historical Total'].apply(lambda x: f"{x:,.0f} L")
                top_historical['Avg Monthly'] = top_historical['Avg Monthly'].apply(lambda x: f"{x:,.0f} L")
                st.dataframe(top_historical, use_container_width=True)
            
            with col2:
                st.write("**Highest Growth Forecast**")
                top_growth = df_rankings.nlargest(10, 'Growth Rate')[['Location', 'Transport Mode', 'Growth Rate', 'Forecast Total']]
                top_growth['Growth Rate'] = top_growth['Growth Rate'].apply(lambda x: f"{x:+.1f}%")
                top_growth['Forecast Total'] = top_growth['Forecast Total'].apply(lambda x: f"{x:,.0f} L")
                st.dataframe(top_growth, use_container_width=True)
            
            st.write("**Complete Rankings**")
            df_display = df_rankings.copy()
            df_display['Historical Total'] = df_display['Historical Total'].apply(lambda x: f"{x:,.0f} L")
            df_display['Avg Monthly'] = df_display['Avg Monthly'].apply(lambda x: f"{x:,.0f} L")
            df_display['Forecast Total'] = df_display['Forecast Total'].apply(lambda x: f"{x:,.0f} L")
            df_display['Growth Rate'] = df_display['Growth Rate'].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(df_display, use_container_width=True)
    
    with tab4:
        st.subheader("Executive Summary")
        
        total_historical = 0
        total_forecast = 0
        mode_summary = {}
        
        for loc in locations:
            if loc in st.session_state.transport_data:
                loc_data = st.session_state.transport_data[loc]
                
                for mode_name, consumption_data in loc_data['modes'].items():
                    historical_total = sum(consumption_data)
                    total_historical += historical_total
                    
                    if mode_name not in mode_summary:
                        mode_summary[mode_name] = {'historical': 0, 'forecast': 0}
                    
                    mode_summary[mode_name]['historical'] += historical_total
                    
                    if (loc, mode_name) in st.session_state.analysis_results:
                        forecast_result = st.session_state.analysis_results[(loc, mode_name)]
                        forecast_total_mode = sum(forecast_result['forecast'])
                        total_forecast += forecast_total_mode
                        mode_summary[mode_name]['forecast'] += forecast_total_mode
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Historical Consumption", 
                f"{total_historical:,.0f} L",
                f"Avg: {total_historical/(num_months*len(locations)):,.0f} L/month/location"
            )
        
        with col2:
            st.metric(
                "Total Forecast Consumption", 
                f"{total_forecast:,.0f} L",
                f"Growth: {((total_forecast/st.session_state.forecast_months)/(total_historical/num_months)-1)*100:+.1f}%"
            )
        
        with col3:
            active_modes = len([k for k, v in mode_summary.items() if v['historical'] > 0])
            st.metric(
                "Active Transport Modes", 
                f"{active_modes}",
                f"Total: {len(TRANSPORT_MODES)} available"
            )
        
        st.write("**Transport Mode Performance Summary:**")
        summary_data = []
        for mode_name, data in mode_summary.items():
            if data['historical'] > 0:
                historical_pct = (data['historical'] / total_historical) * 100
                forecast_pct = (data['forecast'] / total_forecast) * 100 if total_forecast > 0 else 0
                growth = ((data['forecast']/st.session_state.forecast_months) / (data['historical']/num_months) - 1) * 100 if data['historical'] > 0 else 0
                
                summary_data.append({
                    'Transport Mode': mode_name,
                    'Historical Total': f"{data['historical']:,.0f} L",
                    'Historical %': f"{historical_pct:.1f}%",
                    'Forecast Total': f"{data['forecast']:,.0f} L",
                    'Forecast %': f"{forecast_pct:.1f}%",
                    'Growth Rate': f"{growth:+.1f}%"
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            csv = df_summary.to_csv(index=False)
            st.download_button(
                "📥 Download Summary",
                csv,
                "transport_mode_summary.csv",
                "text/csv"
            )

elif st.session_state.transport_data:
    st.header("📊 Data Preview")
    
    preview_data = []
    for loc, loc_data in st.session_state.transport_data.items():
        for mode_name, consumption_data in loc_data['modes'].items():
            if sum(consumption_data) > 0:
                for i, (date, consumption) in enumerate(zip(loc_data['dates'], consumption_data)):
                    preview_data.append({
                        'Date': date.strftime('%b %Y'),
                        'Location': loc,
                        'Transport Mode': mode_name,
                        'Consumption': f"{consumption:,.0f} L"
                    })
    
    if preview_data:
        df_preview = pd.DataFrame(preview_data)
        st.dataframe(df_preview, use_container_width=True)
        
        st.info("👈 Click 'Run Analysis' to start ARIMA forecasting")

else:
    st.info("👈 Setup transport modes in sidebar to begin")
    
    st.subheader("Available Transport Modes:")
    
    mode_info = []
    for mode_name, config in TRANSPORT_MODES.items():
        mode_info.append({
            'Transport Mode': mode_name,
            'Base Consumption': f"{config['base_consumption']} L/day/unit",
            'Efficiency Factor': f"{config['efficiency']}x"
        })
    
    df_modes = pd.DataFrame(mode_info)
    st.dataframe(df_modes, use_container_width=True)

st.markdown("---")
if st.session_state.get('analysis_results'):
    total_models = len(st.session_state.analysis_results)
    st.markdown(f"**✅ Analysis Complete!** | {total_models} transport mode models analyzed")
else:
    st.markdown("**⏳ Ready for Transport Mode Analysis** | Setup data to begin")