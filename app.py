import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Energy Consumption Forecasting",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F5C99;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1F5C99;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1F5C99;
        border-bottom: 2px solid #1F5C99;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    .winner-badge {
        background: #1D9E75;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Load all assets (cached) ──────────────────────────────────
@st.cache_resource
def load_all_assets():
    # Scalers
    with open('models/scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('models/scaler_Y.pkl', 'rb') as f:
        scaler_Y = pickle.load(f)
    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    # Models
    with open('models/ridge_model.pkl', 'rb') as f:
        ridge = pickle.load(f)
    with open('models/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    lstm_model = load_model('models/lstm_model.keras')

    # LSTM config
    with open('models/lstm_config.pkl', 'rb') as f:
        lstm_config = pickle.load(f)

    # Results
    with open('outputs/results_full.csv', 'r') as f:
        results_df = pd.read_csv(f, index_col=0)

    return (scaler_X, scaler_Y, feature_cols,
            ridge, rf_model, lstm_model,
            lstm_config, results_df)


@st.cache_data
def load_data():
    train_df = pd.read_csv('data/processed/train.csv')
    test_df  = pd.read_csv('data/processed/test.csv')
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    return df, train_df, test_df


def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(window_size, len(X)):
        Xs.append(X[i - window_size:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ── Load everything ───────────────────────────────────────────
(scaler_X, scaler_Y, feature_cols,
 ridge, rf_model, lstm_model,
 lstm_config, results_df) = load_all_assets()

df, train_df, test_df = load_data()

X_test_2d = scaler_X.transform(test_df[feature_cols])
y_test_2d = scaler_Y.transform(test_df[['Appliances']])
WINDOW_SIZE = lstm_config['window_size']


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("logo2.png",
             width=60)
    st.markdown("## Energy Forecasting")
    st.markdown("**CS-245 ML + DS-401 Project**")
    st.markdown("---")

    st.markdown("### 🔧 Settings")
    selected_model = st.selectbox(
        "Select Model",
        ["Random Forest", "LSTM", "Ridge Regression", "All Models"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📅 Date Range (Test Set)")
    test_dates = pd.to_datetime(test_df['date'])
    min_date = test_dates.min().date()
    max_date = test_dates.max().date()

    date_start = st.date_input("From", value=min_date,
                                min_value=min_date, max_value=max_date)
    date_end   = st.date_input("To",   value=max_date,
                                min_value=min_date, max_value=max_date)

    n_points = st.slider("Points to display", 100, 1000, 400, 50)

    st.markdown("---")
    st.markdown("### 📊 Display Options")
    show_residuals   = st.checkbox("Show Residual Plot",    value=True)
    show_cluster     = st.checkbox("Show Cluster Analysis", value=True)
    show_feature_imp = st.checkbox("Show Feature Importance (RF)", value=True)

    st.markdown("---")
    st.markdown("**Dataset:** UCI Appliances Energy")
    st.markdown("**Samples:** 19,735 | **Interval:** 10 min")


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">⚡ Appliances Energy Consumption Forecasting</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">UCI Dataset · Machine Learning & Data Science Project · CS-245 + DS-401</div>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🔮 Predictions",
    "📊 Model Comparison",
    "🔍 EDA Insights",
    "📋 Project Info"
])


# ───────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Dataset Overview</div>',
                unsafe_allow_html=True)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",    f"{len(df):,}")
    col2.metric("Avg Energy (Wh)",  f"{df['Appliances'].mean():.1f}")
    col3.metric("Max Energy (Wh)",  f"{df['Appliances'].max():.1f}")
    col4.metric("Features Used",    f"{len(feature_cols)}")

    st.markdown("---")

    # Full time-series
    st.markdown('<div class="section-title">Full Energy Consumption Timeline</div>',
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['Appliances'],
        mode='lines', name='Appliances (Wh)',
        line=dict(color='#1F5C99', width=0.8),
        fill='tozeroy', fillcolor='rgba(31,92,153,0.08)'
    ))
    fig.add_vrect(
        x0=train_df['date'].max(), x1=test_df['date'].max(),
        fillcolor='rgba(29,158,117,0.08)',
        annotation_text="Test Period",
        annotation_position="top left",
        line_width=0
    )
    fig.update_layout(
        title='Energy Consumption Over Time',
        xaxis_title='Date', yaxis_title='Energy (Wh)',
        height=350, showlegend=True,
        plot_bgcolor='white', paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Hourly & weekday patterns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Avg Energy by Hour</div>',
                    unsafe_allow_html=True)
        hourly = df.groupby(df['date'].dt.hour)['Appliances'].mean()
        fig2 = go.Figure(go.Scatter(
            x=hourly.index, y=hourly.values,
            mode='lines+markers',
            line=dict(color='#1F5C99', width=2),
            fill='tozeroy', fillcolor='rgba(31,92,153,0.1)'
        ))
        fig2.update_layout(height=300,
                           xaxis_title='Hour', yaxis_title='Avg Energy (Wh)',
                           plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Avg Energy by Day of Week</div>',
                    unsafe_allow_html=True)
        days  = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        daily = df.groupby(df['date'].dt.dayofweek)['Appliances'].mean()
        colors_dow = ['#E05C4B' if i >= 5 else '#1F5C99' for i in range(7)]
        fig3 = go.Figure(go.Bar(
            x=days, y=daily.values, marker_color=colors_dow, opacity=0.85
        ))
        fig3.update_layout(height=300,
                           xaxis_title='Day', yaxis_title='Avg Energy (Wh)',
                           plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig3, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 2 — PREDICTIONS
# ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Model Predictions vs Actual</div>',
                unsafe_allow_html=True)

    # Filter by date range
    mask = ((pd.to_datetime(test_df['date']).dt.date >= date_start) &
            (pd.to_datetime(test_df['date']).dt.date <= date_end))
    filtered_test = test_df[mask].copy()

    if len(filtered_test) < 50:
        st.warning("Date range too small — please select a wider range.")
    else:
        X_filtered = scaler_X.transform(filtered_test[feature_cols])
        y_filtered = scaler_Y.transform(filtered_test[['Appliances']])
        y_true_f   = scaler_Y.inverse_transform(y_filtered).ravel()

        # Generate predictions based on selection
        model_colors = {
            'Random Forest':    '#1F5C99',
            'LSTM':             '#1D9E75',
            'Ridge Regression': '#E07B39'
        }

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=filtered_test['date'].values[:n_points],
            y=y_true_f[:n_points],
            mode='lines', name='Actual',
            line=dict(color='#2E2E2E', width=1.5)
        ))

        models_to_run = (list(model_colors.keys())
                         if selected_model == "All Models"
                         else [selected_model])

        metrics_live = {}
        for m_name in models_to_run:
            if m_name == 'Ridge Regression':
                y_pred_s = ridge.predict(X_filtered)
                y_pred   = scaler_Y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            elif m_name == 'Random Forest':
                y_pred_s = rf_model.predict(X_filtered)
                y_pred   = scaler_Y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            else:  # LSTM
                if len(X_filtered) <= WINDOW_SIZE:
                    st.warning("Not enough data points for LSTM window. Select wider range.")
                    continue
                X_seq, y_seq = create_sequences(X_filtered, y_filtered, WINDOW_SIZE)
                y_pred_s = lstm_model.predict(X_seq, verbose=0)
                y_pred   = scaler_Y.inverse_transform(y_pred_s).ravel()
                y_true_f_lstm = scaler_Y.inverse_transform(y_seq).ravel()
                rmse = np.sqrt(mean_squared_error(y_true_f_lstm, y_pred))
                mae  = mean_absolute_error(y_true_f_lstm, y_pred)
                metrics_live[m_name] = {'RMSE': rmse, 'MAE': mae}
                fig_pred.add_trace(go.Scatter(
                    x=filtered_test['date'].values[WINDOW_SIZE:WINDOW_SIZE + n_points],
                    y=y_pred[:n_points],
                    mode='lines', name=m_name,
                    line=dict(color=model_colors[m_name], width=1.2, dash='dash')
                ))
                continue

            rmse = np.sqrt(mean_squared_error(y_true_f[:len(y_pred)], y_pred[:len(y_true_f)]))
            mae  = mean_absolute_error(y_true_f[:len(y_pred)], y_pred[:len(y_true_f)])
            metrics_live[m_name] = {'RMSE': rmse, 'MAE': mae}

            fig_pred.add_trace(go.Scatter(
                x=filtered_test['date'].values[:n_points],
                y=y_pred[:n_points],
                mode='lines', name=m_name,
                line=dict(color=model_colors[m_name], width=1.2, dash='dash')
            ))

        fig_pred.update_layout(
            title=f'Predicted vs Actual — {selected_model}',
            xaxis_title='Date', yaxis_title='Energy (Wh)',
            height=420, plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Live metrics
        if metrics_live:
            st.markdown('<div class="section-title">Live Metrics (Selected Range)</div>',
                        unsafe_allow_html=True)
            cols = st.columns(len(metrics_live))
            for col, (m_name, m_vals) in zip(cols, metrics_live.items()):
                col.metric(f"{m_name} — RMSE", f"{m_vals['RMSE']:.2f} Wh")
                col.metric(f"{m_name} — MAE",  f"{m_vals['MAE']:.2f} Wh")

        # Residual plot
        if show_residuals and 'Random Forest' in metrics_live:
            st.markdown('<div class="section-title">Residual Plot</div>',
                        unsafe_allow_html=True)
            y_pred_rf_f = scaler_Y.inverse_transform(
                rf_model.predict(X_filtered).reshape(-1, 1)).ravel()
            residuals = y_true_f - y_pred_rf_f

            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                y=residuals[:n_points], mode='lines',
                line=dict(color='#1F5C99', width=0.8), name='Residuals'
            ))
            fig_res.add_hline(y=0, line_dash='dash', line_color='red')
            fig_res.update_layout(
                title='Random Forest Residuals',
                yaxis_title='Residual (Wh)', height=280,
                plot_bgcolor='white', paper_bgcolor='white'
            )
            st.plotly_chart(fig_res, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 3 — MODEL COMPARISON
# ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Full Model Comparison</div>',
                unsafe_allow_html=True)

    # Metrics table
    st.dataframe(
        results_df.style
            .highlight_min(subset=['RMSE','MAE','MAPE'], color='#d4edda')
            .highlight_max(subset=['R2'],                 color='#d4edda')
            .format({'RMSE': '{:.2f}', 'MAE': '{:.2f}',
                     'MAPE': '{:.2f}%', 'R2': '{:.4f}'}),
        use_container_width=True
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">RMSE Comparison</div>',
                    unsafe_allow_html=True)
        fig_rmse = go.Figure(go.Bar(
            x=results_df.index,
            y=results_df['RMSE'],
            marker_color=['#E07B39', '#1F5C99', '#1D9E75'],
            opacity=0.85, text=results_df['RMSE'].round(2),
            textposition='outside'
        ))
        fig_rmse.update_layout(height=350, yaxis_title='RMSE (Wh)',
                                plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">R² Score Comparison</div>',
                    unsafe_allow_html=True)
        fig_r2 = go.Figure(go.Bar(
            x=results_df.index,
            y=results_df['R2'],
            marker_color=['#E07B39', '#1F5C99', '#1D9E75'],
            opacity=0.85, text=results_df['R2'].round(4),
            textposition='outside'
        ))
        fig_r2.update_layout(height=350, yaxis_title='R² Score',
                              plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_r2, use_container_width=True)

    # Feature importance
    if show_feature_imp:
        st.markdown('<div class="section-title">Random Forest — Feature Importance</div>',
                    unsafe_allow_html=True)
        importances = rf_model.feature_importances_
        indices     = np.argsort(importances)[::-1][:15]
        top_feats   = [feature_cols[i] for i in indices]
        top_imps    = importances[indices]

        fig_imp = go.Figure(go.Bar(
            x=top_imps[::-1], y=top_feats[::-1],
            orientation='h',
            marker_color=['#1F5C99' if i == len(top_feats)-1
                          else '#D6E4F0' for i in range(len(top_feats))],
            opacity=0.9
        ))
        fig_imp.update_layout(
            height=450, xaxis_title='Importance Score',
            plot_bgcolor='white', paper_bgcolor='white'
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 4 — EDA INSIGHTS
# ───────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    # Heatmap hour vs weekday
    st.markdown('<div class="section-title">Energy Heatmap — Hour vs Day of Week</div>',
                unsafe_allow_html=True)
    df['hour']       = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    pivot = df.pivot_table(values='Appliances',
                           index='day_of_week', columns='hour', aggfunc='mean')
    pivot.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

    fig_heat = px.imshow(
        pivot, color_continuous_scale='YlOrRd',
        labels=dict(x='Hour of Day', y='Day of Week', color='Avg Energy (Wh)'),
        title='Average Energy Consumption Heatmap'
    )
    fig_heat.update_layout(height=350)
    st.plotly_chart(fig_heat, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Distribution
        st.markdown('<div class="section-title">Energy Distribution</div>',
                    unsafe_allow_html=True)
        fig_dist = px.histogram(df, x='Appliances', nbins=80,
                                color_discrete_sequence=['#1F5C99'],
                                title='Distribution of Appliance Energy (Wh)')
        fig_dist.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Cluster analysis
        if show_cluster and 'cluster' in df.columns:
            st.markdown('<div class="section-title">K-Means Cluster Distribution</div>',
                        unsafe_allow_html=True)
            cluster_map = {0: 'Low Usage', 1: 'Medium Usage', 2: 'High Usage'}
            df['cluster_label'] = df['cluster'].map(cluster_map)
            cluster_counts = df['cluster_label'].value_counts()
            fig_pie = px.pie(
                values=cluster_counts.values,
                names=cluster_counts.index,
                color_discrete_sequence=['#1D9E75','#1F5C99','#E05C4B'],
                title='Cluster Distribution'
            )
            fig_pie.update_layout(height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

    # Correlation bar
    st.markdown('<div class="section-title">Top Feature Correlations with Target</div>',
                unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()['Appliances'].drop('Appliances').sort_values(key=abs, ascending=False)
    colors_corr = ['#1D9E75' if v > 0 else '#E05C4B' for v in corr.values[:15]]

    fig_corr = go.Figure(go.Bar(
        x=corr.values[:15], y=corr.index[:15],
        orientation='h', marker_color=colors_corr, opacity=0.85
    ))
    fig_corr.update_layout(
        height=400, xaxis_title='Correlation Coefficient',
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 5 — PROJECT INFO
# ───────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">Project Information</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **📚 Courses**
        - CS-245: Machine Learning
        - DS-401: Introduction to Data Science

        **📦 Dataset**
        - UCI Appliances Energy Prediction
        - ID: 374 | License: CC BY 4.0
        - 19,735 rows | 10-min intervals | 4.5 months

        **🧠 Models Implemented**
        - Ridge Regression (baseline)
        -  Random Forest Regressor
        -  LSTM Neural Network
        -  K-Means Clustering (unsupervised)
        """)

    with col2:
        st.markdown("""
        **🛠️ Tools & Libraries**
        - Python, Pandas, NumPy
        - Scikit-learn, TensorFlow/Keras
        - Matplotlib, Seaborn, Plotly
        - Streamlit

        **📊 Evaluation Metrics**
        - RMSE — Root Mean Squared Error
        - MAE  — Mean Absolute Error
        - MAPE — Mean Absolute Percentage Error
        - R²   — Coefficient of Determination

        **🔗 Dataset Link**
        - https://archive.ics.uci.edu/dataset/374
        """)

    st.markdown("---")
    st.markdown('<div class="section-title">Critical Reflection</div>',
                unsafe_allow_html=True)
    try:
        with open('outputs/critical_reflection.txt', 'r',encoding='utf-8', errors='ignore') as f:
            reflection = f.read()
        st.code(reflection, language=None)
    except Exception as e:
        st.error(f"Error: {e}")
    st.markdown("---")
    st.markdown('<div class="section-title">Full Results Table</div>',
                unsafe_allow_html=True)
    st.dataframe(results_df, use_container_width=True)