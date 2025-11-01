import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
CONFIG = {
    'occupation_impact_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\occupation_ai_impact_final.csv",
    'output_html': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\ai_impact_dashboard.html",
    'years': list(range(2024, 2036))
}

def create_dashboard():
    """Create comprehensive AI impact dashboard"""
    
    # Load data
    df = pd.read_csv(CONFIG['occupation_impact_path'])
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Top 20 Most Vulnerable Occupations (2035)',
            'Vulnerability Distribution Over Time',
            'Vulnerability Trajectory (Top 10 Occupations)',
            'Uncertainty Bands (Sample Occupation)',
            'Vulnerability Heatmap (Top 15 Occupations)',
            'Risk Categories Over Time'
        ),
        specs=[
            [{"type": "bar"}, {"type": "violin"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # --- PLOT 1: Top Vulnerable Occupations ---
    top_20 = df.nlargest(20, 'Vulnerability_Median_2035').sort_values('Vulnerability_Median_2035')
    
    fig.add_trace(
        go.Bar(
            y=top_20['Occupation'],
            x=top_20['Vulnerability_Median_2035'],
            orientation='h',
            marker=dict(color=top_20['Vulnerability_Median_2035'], colorscale='Reds'),
            name='Vulnerability'
        ),
        row=1, col=1
    )
    
    # --- PLOT 2: Distribution Over Time ---
    for year in [2024, 2028, 2032, 2035]:
        fig.add_trace(
            go.Violin(
                y=df[f'Vulnerability_Median_{year}'],
                name=str(year),
                box_visible=True,
                meanline_visible=True
            ),
            row=1, col=2
        )
    
    # --- PLOT 3: Trajectory Lines ---
    top_10 = df.nlargest(10, 'Vulnerability_Median_2035')
    
    for _, occ in top_10.iterrows():
        values = [occ[f'Vulnerability_Median_{year}'] for year in CONFIG['years']]
        fig.add_trace(
            go.Scatter(
                x=CONFIG['years'],
                y=values,
                mode='lines+markers',
                name=occ['Occupation'][:30],  # Truncate long names
                line=dict(width=2)
            ),
            row=2, col=1
        )
    
    # --- PLOT 4: Uncertainty Bands ---
    sample_occ = df.nlargest(1, 'Vulnerability_Median_2035').iloc[0]
    
    median_vals = [sample_occ[f'Vulnerability_Median_{year}'] for year in CONFIG['years']]
    p05_vals = [sample_occ[f'Vulnerability_p05_{year}'] for year in CONFIG['years']]
    p95_vals = [sample_occ[f'Vulnerability_p95_{year}'] for year in CONFIG['years']]
    
    # 95th percentile
    fig.add_trace(
        go.Scatter(
            x=CONFIG['years'],
            y=p95_vals,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=2
    )
    
    # Fill area
    fig.add_trace(
        go.Scatter(
            x=CONFIG['years'],
            y=p05_vals,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            name='90% Confidence',
            hoverinfo='skip'
        ),
        row=2, col=2
    )
    
    # Median
    fig.add_trace(
        go.Scatter(
            x=CONFIG['years'],
            y=median_vals,
            mode='lines+markers',
            line=dict(color='red', width=3),
            name=f'{sample_occ["Occupation"][:40]}'
        ),
        row=2, col=2
    )
    
    # --- PLOT 5: Heatmap ---
    top_15 = df.nlargest(15, 'Vulnerability_Median_2035')
    heatmap_data = []
    
    for _, occ in top_15.iterrows():
        row_data = [occ[f'Vulnerability_Median_{year}'] for year in CONFIG['years']]
        heatmap_data.append(row_data)
    
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=CONFIG['years'],
            y=[occ[:40] for occ in top_15['Occupation']],
            colorscale='Reds',
            colorbar=dict(x=1.15)
        ),
        row=3, col=1
    )
    
    # --- PLOT 6: Risk Categories ---
    risk_categories = []
    
    for year in CONFIG['years']:
        high_risk = (df[f'Vulnerability_Median_{year}'] > 0.7).sum()
        medium_risk = ((df[f'Vulnerability_Median_{year}'] > 0.4) & 
                       (df[f'Vulnerability_Median_{year}'] <= 0.7)).sum()
        low_risk = (df[f'Vulnerability_Median_{year}'] <= 0.4).sum()
        
        risk_categories.append({
            'Year': year,
            'High Risk (>0.7)': high_risk,
            'Medium Risk (0.4-0.7)': medium_risk,
            'Low Risk (<0.4)': low_risk
        })
    
    risk_df = pd.DataFrame(risk_categories)
    
    for category in ['High Risk (>0.7)', 'Medium Risk (0.4-0.7)', 'Low Risk (<0.4)']:
        fig.add_trace(
            go.Bar(
                x=risk_df['Year'],
                y=risk_df[category],
                name=category
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=1400,
        width=1600,
        title_text="<b>AI Occupational Impact Dashboard (2024-2035)</b>",
        title_font_size=24,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Vulnerability Score", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_xaxes(title_text="Year", row=3, col=2)
    
    fig.update_yaxes(title_text="Occupation", row=1, col=1)
    fig.update_yaxes(title_text="Vulnerability", row=1, col=2)
    fig.update_yaxes(title_text="Vulnerability", row=2, col=1)
    fig.update_yaxes(title_text="Vulnerability", row=2, col=2)
    fig.update_yaxes(title_text="Number of Occupations", row=3, col=2)
    
    # Save
    fig.write_html(CONFIG['output_html'])
    print(f"✅ Dashboard saved to {CONFIG['output_html']}")
    
    return fig

if __name__ == "__main__":
    create_dashboard()