"""
Hospital Operations Center - Analytics Dashboard
================================================
A meta-dashboard for Administrators and Data Scientists to monitor:
- Hospital flow and patient throughput
- AI model performance and drift
- SHAP explainability aggregates
- Real-time operational metrics

Run with: streamlit run src/analytics/dashboard.py
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Database URL - prefer environment variable, fallback to SQLite
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./triage_dev.db"
)

# Normalize postgres:// to postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Page config
st.set_page_config(
    page_title="🏥 Hospital Operations Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Database Connection
# -----------------------------------------------------------------------------

@st.cache_resource
def get_engine():
    """Create SQLAlchemy engine with connection pooling."""
    connect_args = {}
    if DATABASE_URL.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    
    return create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        pool_pre_ping=True if not DATABASE_URL.startswith("sqlite") else False,
    )


# -----------------------------------------------------------------------------
# Data Loading Functions
# -----------------------------------------------------------------------------

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_triage_data(hours: int = 24) -> pd.DataFrame:
    """
    Load triage results and appointment data for analytics.
    
    Returns DataFrame with:
    - timestamp, triage_level, shock_index, wait_time, rl_action
    - vitals data for clinical analysis
    """
    engine = get_engine()
    
    from datetime import timezone
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    query = text("""
        SELECT 
            tr.id,
            tr.appointment_id,
            tr.esi_level as triage_level,
            tr.supervised_confidence as confidence,
            tr.rl_action,
            tr.shap_values,
            tr.vitals,
            tr.created_at as timestamp,
            a.scheduled_time as arrival_time,
            a.status,
            pt.token_number,
            pt.priority_score,
            pt.estimated_wait_minutes
        FROM triage_results tr
        JOIN appointments a ON tr.appointment_id = a.id
        LEFT JOIN priority_tokens pt ON a.id = pt.appointment_id
        WHERE tr.created_at >= :cutoff
        ORDER BY tr.created_at DESC
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"cutoff": cutoff})
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return pd.DataFrame()
    
    if df.empty:
        return df
    
    # Parse vitals JSON and calculate Shock Index
    import json
    
    def parse_vitals(vitals_json):
        if pd.isna(vitals_json) or vitals_json is None:
            return {}
        if isinstance(vitals_json, str):
            try:
                parsed = json.loads(vitals_json)
                return parsed if isinstance(parsed, dict) else {}
            except:
                return {}
        return vitals_json if isinstance(vitals_json, dict) else {}
    
    def safe_get(d, *keys):
        """Safely get value from dict, handling nested strings."""
        if not isinstance(d, dict):
            return None
        for key in keys:
            if d.get(key) is not None:
                return d.get(key)
        return None
    
    df['vitals_parsed'] = df['vitals'].apply(parse_vitals)
    df['heart_rate'] = df['vitals_parsed'].apply(lambda x: safe_get(x, 'Pulse', 'HR'))
    df['sbp'] = df['vitals_parsed'].apply(lambda x: safe_get(x, 'SBP'))
    df['o2sat'] = df['vitals_parsed'].apply(lambda x: safe_get(x, 'O2Sat'))
    df['temp'] = df['vitals_parsed'].apply(lambda x: safe_get(x, 'Temp'))
    df['age'] = df['vitals_parsed'].apply(lambda x: safe_get(x, 'Age'))
    
    # Calculate Shock Index
    df['shock_index'] = df.apply(
        lambda row: row['heart_rate'] / row['sbp'] 
        if row['heart_rate'] and row['sbp'] and row['sbp'] > 0 
        else None, 
        axis=1
    )
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    
    return df


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_shap_aggregate(limit: int = 100) -> pd.DataFrame:
    """
    Aggregate SHAP values from recent triage results.
    Returns top features by mean absolute SHAP value.
    """
    engine = get_engine()
    
    query = text("""
        SELECT shap_values
        FROM triage_results
        WHERE shap_values IS NOT NULL
        ORDER BY created_at DESC
        LIMIT :limit
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"limit": limit})
            rows = result.fetchall()
    except Exception as e:
        st.error(f"Error loading SHAP data: {e}")
        return pd.DataFrame()
    
    if not rows:
        return pd.DataFrame()
    
    # Feature names (these should match your preprocessor output)
    feature_names = [
        'Age', 'Temp', 'Pulse', 'Resp', 'SBP', 'DBP', 'O2Sat', 'PainScale',
        'ArrivalMode_Ambulance', 'ArrivalMode_Public', 'ArrivalMode_Walk-in'
    ] + [f'BERT_{i}' for i in range(768)]  # Placeholder for BERT features
    
    # Aggregate SHAP values
    all_shap = []
    import json
    for row in rows:
        shap_json = row[0]
        if shap_json:
            try:
                if isinstance(shap_json, str):
                    shap_data = json.loads(shap_json)
                else:
                    shap_data = shap_json
                
                # Handle different SHAP formats
                if isinstance(shap_data, list) and len(shap_data) > 0:
                    # Multi-class: take mean across classes
                    if isinstance(shap_data[0], list):
                        shap_flat = np.mean(np.array(shap_data), axis=0).flatten()
                    else:
                        shap_flat = np.array(shap_data).flatten()
                    all_shap.append(shap_flat)
            except:
                continue
    
    if not all_shap:
        return pd.DataFrame()
    
    # Calculate mean absolute SHAP per feature
    shap_matrix = np.array(all_shap)
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    
    # Match feature names (truncate if needed)
    n_features = len(mean_abs_shap)
    if n_features <= len(feature_names):
        names = feature_names[:n_features]
    else:
        names = feature_names + [f'Feature_{i}' for i in range(len(feature_names), n_features)]
    
    df = pd.DataFrame({
        'feature': names,
        'importance': mean_abs_shap
    })
    
    return df.nlargest(15, 'importance')


def get_queue_stats() -> dict:
    """Get current queue statistics."""
    engine = get_engine()
    
    query = text("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'SCHEDULED' THEN 1 ELSE 0 END) as waiting,
            SUM(CASE WHEN status = 'IN_PROGRESS' THEN 1 ELSE 0 END) as in_progress,
            SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed
        FROM appointments
        WHERE DATE(created_at) = DATE('now')
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            row = result.fetchone()
            return {
                'total': row[0] or 0,
                'waiting': row[1] or 0,
                'in_progress': row[2] or 0,
                'completed': row[3] or 0
            }
    except:
        return {'total': 0, 'waiting': 0, 'in_progress': 0, 'completed': 0}


# -----------------------------------------------------------------------------
# Visualization Components
# -----------------------------------------------------------------------------

def render_kpi_cards(df: pd.DataFrame, stats: dict):
    """Render KPI metric cards at the top."""
    from datetime import timezone
    col1, col2, col3, col4 = st.columns(4)
    
    today = datetime.now(timezone.utc).date()
    today_df = df[df['date'] == today] if not df.empty else pd.DataFrame()
    
    with col1:
        st.metric(
            label="📊 Total Patients Today",
            value=len(today_df),
            delta=f"+{len(today_df) - stats.get('completed', 0)} pending" if len(today_df) > 0 else None
        )
    
    with col2:
        # Average wait time for high acuity (levels 1-2)
        high_acuity = today_df[today_df['triage_level'].isin([1, 2])]
        avg_wait = high_acuity['estimated_wait_minutes'].mean() if not high_acuity.empty else 0
        st.metric(
            label="⏱️ Avg Wait (High Acuity)",
            value=f"{avg_wait:.0f} min" if avg_wait else "N/A",
            delta=None
        )
    
    with col3:
        occupancy = stats.get('in_progress', 0)
        st.metric(
            label="🛏️ Current Occupancy",
            value=occupancy,
            delta=f"{stats.get('waiting', 0)} waiting"
        )
    
    with col4:
        # AI confidence
        avg_conf = today_df['confidence'].mean() if not today_df.empty else 0
        st.metric(
            label="🤖 Avg AI Confidence",
            value=f"{avg_conf:.1%}" if avg_conf else "N/A",
        )


def render_arrivals_chart(df: pd.DataFrame):
    """Time-series of arrivals per hour."""
    if df.empty:
        st.info("No data available for arrivals chart.")
        return
    
    hourly = df.groupby('hour').size().reset_index(name='count')
    hourly = hourly.sort_values('hour')
    
    # Fill missing hours
    all_hours = pd.DataFrame({'hour': range(24)})
    hourly = all_hours.merge(hourly, on='hour', how='left').fillna(0)
    
    fig = px.line(
        hourly,
        x='hour',
        y='count',
        title='📈 Arrivals per Hour (Last 24h)',
        labels={'hour': 'Hour of Day', 'count': 'Number of Patients'},
        markers=True,
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=2),
        yaxis=dict(rangemode='tozero'),
        hovermode='x unified',
    )
    
    fig.update_traces(
        line=dict(color='#6366f1', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.1)',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_token_distribution(df: pd.DataFrame):
    """Bar chart of token distribution by triage level."""
    if df.empty:
        st.info("No data available for token distribution.")
        return
    
    level_counts = df['triage_level'].value_counts().sort_index()
    
    colors = {
        1: '#ef4444',  # Red
        2: '#f97316',  # Orange
        3: '#eab308',  # Yellow
        4: '#22c55e',  # Green
        5: '#3b82f6',  # Blue
    }
    
    labels = {
        1: 'RED (Resuscitation)',
        2: 'ORANGE (Emergent)',
        3: 'YELLOW (Urgent)',
        4: 'GREEN (Less Urgent)',
        5: 'BLUE (Non-Urgent)',
    }
    
    fig = go.Figure()
    
    for level in sorted(level_counts.index):
        fig.add_trace(go.Bar(
            x=[labels.get(level, f'Level {level}')],
            y=[level_counts[level]],
            name=labels.get(level, f'Level {level}'),
            marker_color=colors.get(level, '#9ca3af'),
            text=[level_counts[level]],
            textposition='outside',
        ))
    
    fig.update_layout(
        title='🎫 Token Distribution by Triage Level',
        xaxis_title='Triage Level',
        yaxis_title='Count',
        showlegend=False,
        bargap=0.3,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_shock_index_scatter(df: pd.DataFrame):
    """Scatter plot of HR vs SBP colored by triage level."""
    plot_df = df.dropna(subset=['heart_rate', 'sbp'])
    
    if plot_df.empty:
        st.info("No vitals data available for Shock Index analysis.")
        return
    
    # Create color map
    color_map = {
        1: '#ef4444',
        2: '#f97316', 
        3: '#eab308',
        4: '#22c55e',
        5: '#3b82f6',
    }
    
    fig = px.scatter(
        plot_df,
        x='sbp',
        y='heart_rate',
        color='triage_level',
        color_discrete_map=color_map,
        title='💓 Shock Index Analysis (HR vs SBP)',
        labels={
            'sbp': 'Systolic Blood Pressure (mmHg)',
            'heart_rate': 'Heart Rate (bpm)',
            'triage_level': 'Triage Level',
        },
        hover_data=['shock_index', 'rl_action', 'token_number'],
    )
    
    # Add SI = 1.0 reference line (HR/SBP = 1 → HR = SBP)
    sbp_range = [plot_df['sbp'].min() - 10, plot_df['sbp'].max() + 10]
    fig.add_trace(go.Scatter(
        x=sbp_range,
        y=sbp_range,  # SI=1 line
        mode='lines',
        name='SI = 1.0 (Critical)',
        line=dict(color='red', dash='dash', width=2),
    ))
    
    # Add SI = 0.7 reference line (normal upper limit)
    fig.add_trace(go.Scatter(
        x=sbp_range,
        y=[x * 0.7 for x in sbp_range],
        mode='lines',
        name='SI = 0.7 (Normal)',
        line=dict(color='green', dash='dot', width=1),
    ))
    
    fig.update_layout(
        xaxis=dict(range=[50, 200]),
        yaxis=dict(range=[40, 180]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_shap_summary(shap_df: pd.DataFrame):
    """Horizontal bar chart of top SHAP features."""
    if shap_df.empty:
        st.info("No SHAP data available. SHAP values are computed in background after triage.")
        return
    
    # Filter out BERT features for cleaner visualization
    non_bert = shap_df[~shap_df['feature'].str.startswith('BERT_')]
    
    if non_bert.empty:
        non_bert = shap_df.head(10)
    
    fig = px.bar(
        non_bert.head(10),
        y='feature',
        x='importance',
        orientation='h',
        title='🧠 Top 10 Features Driving AI Decisions',
        labels={'importance': 'Mean |SHAP Value|', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis',
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_confidence_histogram(df: pd.DataFrame):
    """Histogram of model confidence scores."""
    if df.empty or df['confidence'].isna().all():
        st.info("No confidence data available.")
        return
    
    fig = px.histogram(
        df.dropna(subset=['confidence']),
        x='confidence',
        nbins=20,
        title='📊 Model Confidence Distribution',
        labels={'confidence': 'Confidence Score', 'count': 'Count'},
        color_discrete_sequence=['#6366f1'],
    )
    
    fig.add_vline(
        x=df['confidence'].mean(),
        line_dash='dash',
        line_color='red',
        annotation_text=f"Mean: {df['confidence'].mean():.2f}",
    )
    
    fig.update_layout(
        xaxis=dict(range=[0, 1]),
        bargap=0.1,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_rl_action_analysis(df: pd.DataFrame):
    """Analyze RL actions taken."""
    if df.empty or df['rl_action'].isna().all():
        st.info("No RL action data available.")
        return
    
    action_counts = df['rl_action'].value_counts()
    
    fig = px.pie(
        values=action_counts.values,
        names=action_counts.index,
        title='🎮 RL Action Distribution',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

def main():
    # Header
    st.markdown('<p class="main-header">🏥 Hospital Operations Center</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time Analytics & AI Performance Monitoring</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/hospital-3.png", width=80)
        st.title("Settings")
        
        hours = st.slider("Data Window (hours)", 1, 168, 24)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.markdown("### Database Status")
        try:
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            st.success("✅ Connected")
            st.caption(f"URL: `{DATABASE_URL[:30]}...`")
        except Exception as e:
            st.error(f"❌ Disconnected: {e}")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_triage_data(hours=hours)
        stats = get_queue_stats()
        shap_df = load_shap_aggregate(limit=100)
    
    # KPI Cards
    render_kpi_cards(df, stats)
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🏥 Hospital Flow", "🤖 AI Performance", "📋 Raw Data"])
    
    # Tab 1: Hospital Flow
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            render_arrivals_chart(df)
        
        with col2:
            render_token_distribution(df)
        
        st.divider()
        
        col3, col4 = st.columns(2)
        
        with col3:
            render_rl_action_analysis(df)
        
        with col4:
            # Wait time by triage level
            if not df.empty and 'estimated_wait_minutes' in df.columns:
                wait_by_level = df.groupby('triage_level')['estimated_wait_minutes'].mean().dropna()
                if not wait_by_level.empty:
                    fig = px.bar(
                        x=wait_by_level.index,
                        y=wait_by_level.values,
                        title='⏳ Average Wait Time by Triage Level',
                        labels={'x': 'Triage Level', 'y': 'Minutes'},
                        color=wait_by_level.values,
                        color_continuous_scale='RdYlGn_r',
                    )
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: AI Performance
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            render_shock_index_scatter(df)
        
        with col2:
            render_confidence_histogram(df)
        
        st.divider()
        
        render_shap_summary(shap_df)
        
        # Model drift indicator
        if not df.empty:
            st.subheader("📉 Model Performance Trends")
            
            # Group by date and calculate metrics
            daily_metrics = df.groupby('date').agg({
                'confidence': 'mean',
                'triage_level': 'mean',
                'shock_index': 'mean',
            }).reset_index()
            
            if len(daily_metrics) > 1:
                fig = make_subplots(rows=1, cols=3, subplot_titles=[
                    'Avg Confidence', 'Avg Triage Level', 'Avg Shock Index'
                ])
                
                fig.add_trace(go.Scatter(
                    x=daily_metrics['date'], y=daily_metrics['confidence'],
                    mode='lines+markers', name='Confidence'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=daily_metrics['date'], y=daily_metrics['triage_level'],
                    mode='lines+markers', name='Triage Level'
                ), row=1, col=2)
                
                fig.add_trace(go.Scatter(
                    x=daily_metrics['date'], y=daily_metrics['shock_index'],
                    mode='lines+markers', name='Shock Index'
                ), row=1, col=3)
                
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Raw Data
    with tab3:
        st.subheader("📋 Recent Triage Results")
        
        if not df.empty:
            display_cols = [
                'timestamp', 'token_number', 'triage_level', 'confidence',
                'shock_index', 'rl_action', 'heart_rate', 'sbp', 'o2sat'
            ]
            display_df = df[[c for c in display_cols if c in df.columns]].head(100)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Data (CSV)",
                data=csv,
                file_name=f"triage_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No data available. Run some triage assessments first!")


if __name__ == "__main__":
    main()
