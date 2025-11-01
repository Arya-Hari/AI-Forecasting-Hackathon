import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# --- CONFIGURATION ---
CONFIG = {
    'occupation_impact_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\occupation_ai_impact_final.csv",
    'output_html': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\ai_impact_interactive_dashboard.html",
    'years': list(range(2024, 2036))
}

def create_interactive_dashboard():
    """Create fully interactive AI impact dashboard with sliders and filters"""
    
    print("=" * 60)
    print("CREATING INTERACTIVE AI IMPACT DASHBOARD")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv(CONFIG['occupation_impact_path'])
    print(f"   ✅ Loaded {len(df):,} occupations")
    
    # Create main figure with tabs using Plotly's updatemenus
    # We'll create multiple views that can be toggled
    
    # === PREPARATION ===
    # Sort by 2035 vulnerability for consistent ordering
    df_sorted = df.sort_values('Vulnerability_Median_2035', ascending=False)
    
    # Get top occupations for detailed views
    top_50 = df_sorted.head(50)
    
    print("\n🎨 Building interactive visualizations...")
    
    # ============================================================
    # MAIN INTERACTIVE FIGURE WITH YEAR SLIDER
    # ============================================================
    
    # Create frames for animation/slider (one frame per year)
    frames = []
    
    for year in CONFIG['years']:
        # Data for this year
        year_data = df.nlargest(30, f'Vulnerability_Median_{year}').sort_values(f'Vulnerability_Median_{year}')
        
        frame_data = [
            go.Bar(
                y=year_data['Occupation'],
                x=year_data[f'Vulnerability_Median_{year}'],
                orientation='h',
                marker=dict(
                    color=year_data[f'Vulnerability_Median_{year}'],
                    colorscale='Reds',
                    cmin=0,
                    cmax=1
                ),
                text=year_data[f'Vulnerability_Median_{year}'].round(3),
                textposition='outside',
                name=f'Vulnerability {year}',
                hovertemplate='<b>%{y}</b><br>Vulnerability: %{x:.3f}<extra></extra>'
            )
        ]
        
        frames.append(go.Frame(data=frame_data, name=str(year)))
    
    # Initial frame (2024)
    initial_data = df.nlargest(30, 'Vulnerability_Median_2024').sort_values('Vulnerability_Median_2024')
    
    fig = go.Figure(
        data=[
            go.Bar(
                y=initial_data['Occupation'],
                x=initial_data['Vulnerability_Median_2024'],
                orientation='h',
                marker=dict(
                    color=initial_data['Vulnerability_Median_2024'],
                    colorscale='Reds',
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="Vulnerability<br>Score", x=1.02)
                ),
                text=initial_data['Vulnerability_Median_2024'].round(3),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Vulnerability: %{x:.3f}<extra></extra>'
            )
        ],
        frames=frames
    )
    
    # Add year slider
    sliders = [{
        'active': 0,
        'yanchor': 'top',
        'y': -0.15,
        'xanchor': 'left',
        'currentvalue': {
            'prefix': 'Year: ',
            'visible': True,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#666'}
        },
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.05,
        'steps': [
            {
                'args': [[str(year)], {
                    'frame': {'duration': 300, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 300}
                }],
                'label': str(year),
                'method': 'animate'
            }
            for year in CONFIG['years']
        ]
    }]
    
    # Add play/pause buttons
    updatemenus = [{
        'type': 'buttons',
        'showactive': False,
        'y': -0.15,
        'x': -0.05,
        'xanchor': 'right',
        'yanchor': 'top',
        'buttons': [
            {
                'label': '▶ Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 500, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 300}
                }]
            },
            {
                'label': '⏸ Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
        ]
    }]
    
    fig.update_layout(
        title={
            'text': '<b>Top 30 Most Vulnerable Occupations by Year</b><br><sub>Use slider to explore different years | Click Play to animate</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        xaxis_title='<b>Vulnerability Score</b>',
        yaxis_title='',
        height=900,
        width=1400,
        template='plotly_white',
        sliders=sliders,
        updatemenus=updatemenus,
        xaxis=dict(range=[0, 1.0]),
        margin=dict(l=300, r=100, t=120, b=150)
    )
    
    print("   ✅ Created animated bar chart with year slider")
    
    # Save first figure
    fig.write_html(CONFIG['output_html'])
    print(f"\n💾 Main dashboard saved to {CONFIG['output_html']}")
    
    # ============================================================
    # CREATE MULTI-TAB DASHBOARD WITH DROPDOWNS
    # ============================================================
    
    print("\n🎨 Creating comprehensive multi-view dashboard...")
    
    # Create a more comprehensive dashboard with multiple interactive elements
    from plotly.subplots import make_subplots
    
    # Prepare data for different views
    risk_categories_over_time = []
    for year in CONFIG['years']:
        high = (df[f'Vulnerability_Median_{year}'] > 0.7).sum()
        medium = ((df[f'Vulnerability_Median_{year}'] > 0.4) & (df[f'Vulnerability_Median_{year}'] <= 0.7)).sum()
        low = (df[f'Vulnerability_Median_{year}'] <= 0.4).sum()
        risk_categories_over_time.append({
            'Year': year,
            'High Risk (>0.7)': high,
            'Medium Risk (0.4-0.7)': medium,
            'Low Risk (<0.4)': low
        })
    risk_df = pd.DataFrame(risk_categories_over_time)
    
    # Create subplot figure
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Vulnerability Distribution Over Time',
            'Top 10 Occupations - Trajectory with Uncertainty',
            'Risk Category Evolution',
            'Heatmap: Top 20 Occupations Over Time'
        ),
        specs=[
            [{"type": "box"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "heatmap"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # --- PLOT 1: Box plots over time ---
    for year in CONFIG['years']:
        fig2.add_trace(
            go.Box(
                y=df[f'Vulnerability_Median_{year}'],
                name=str(year),
                marker_color='indianred',
                boxmean='sd'
            ),
            row=1, col=1
        )
    
    # --- PLOT 2: Top 10 trajectories with uncertainty bands ---
    top_10 = df.nlargest(10, 'Vulnerability_Median_2035')
    
    colors = px.colors.qualitative.Set3[:10]
    
    for idx, (_, occ) in enumerate(top_10.iterrows()):
        occ_name = occ['Occupation'][:40]  # Truncate
        
        # Get values
        median_vals = [occ[f'Vulnerability_Median_{year}'] for year in CONFIG['years']]
        p05_vals = [occ[f'Vulnerability_p05_{year}'] for year in CONFIG['years']]
        p95_vals = [occ[f'Vulnerability_p95_{year}'] for year in CONFIG['years']]
        
        # Add uncertainty band
        fig2.add_trace(
            go.Scatter(
                x=CONFIG['years'] + CONFIG['years'][::-1],
                y=p95_vals + p05_vals[::-1],
                fill='toself',
                fillcolor=colors[idx],
                opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                name=occ_name
            ),
            row=1, col=2
        )
        
        # Add median line
        fig2.add_trace(
            go.Scatter(
                x=CONFIG['years'],
                y=median_vals,
                mode='lines+markers',
                name=occ_name,
                line=dict(color=colors[idx], width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{occ_name}</b><br>Year: %{{x}}<br>Vulnerability: %{{y:.3f}}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # --- PLOT 3: Stacked bar chart of risk categories ---
    fig2.add_trace(
        go.Bar(
            x=risk_df['Year'],
            y=risk_df['High Risk (>0.7)'],
            name='High Risk (>0.7)',
            marker_color='darkred',
            hovertemplate='Year: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig2.add_trace(
        go.Bar(
            x=risk_df['Year'],
            y=risk_df['Medium Risk (0.4-0.7)'],
            name='Medium Risk (0.4-0.7)',
            marker_color='orange',
            hovertemplate='Year: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig2.add_trace(
        go.Bar(
            x=risk_df['Year'],
            y=risk_df['Low Risk (<0.4)'],
            name='Low Risk (<0.4)',
            marker_color='lightgreen',
            hovertemplate='Year: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # --- PLOT 4: Heatmap ---
    top_20_heat = df.nlargest(20, 'Vulnerability_Median_2035')
    heatmap_data = []
    y_labels = []
    
    for _, occ in top_20_heat.iterrows():
        row_data = [occ[f'Vulnerability_Median_{year}'] for year in CONFIG['years']]
        heatmap_data.append(row_data)
        y_labels.append(occ['Occupation'][:45])
    
    fig2.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=CONFIG['years'],
            y=y_labels,
            colorscale='Reds',
            colorbar=dict(title="Vulnerability", x=1.15),
            hovertemplate='<b>%{y}</b><br>Year: %{x}<br>Vulnerability: %{z:.3f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig2.update_layout(
        height=1000,
        width=1600,
        title_text="<b>AI Occupational Impact - Comprehensive Analysis Dashboard</b>",
        title_font_size=22,
        showlegend=True,
        template='plotly_white',
        barmode='stack',
        hovermode='closest'
    )
    
    # Update axes
    fig2.update_xaxes(title_text="<b>Year</b>", row=1, col=1, tickangle=45)
    fig2.update_xaxes(title_text="<b>Year</b>", row=1, col=2)
    fig2.update_xaxes(title_text="<b>Year</b>", row=2, col=1)
    fig2.update_xaxes(title_text="<b>Year</b>", row=2, col=2)
    
    fig2.update_yaxes(title_text="<b>Vulnerability Score</b>", row=1, col=1)
    fig2.update_yaxes(title_text="<b>Vulnerability Score</b>", row=1, col=2)
    fig2.update_yaxes(title_text="<b>Number of Occupations</b>", row=2, col=1)
    
    # Save second dashboard
    output_path_2 = CONFIG['output_html'].replace('.html', '_comprehensive.html')
    fig2.write_html(output_path_2)
    print(f"   ✅ Comprehensive dashboard saved to {output_path_2}")
    
    # ============================================================
    # CREATE SCATTER PLOT WITH RANGE SLIDERS
    # ============================================================
    
    print("\n🎨 Creating interactive scatter plot explorer...")
    
    # Create scatter plot for 2035 with range sliders
    fig3 = go.Figure()
    
    # Add scatter trace
    fig3.add_trace(
        go.Scatter(
            x=df['Vulnerability_Median_2024'],
            y=df['Vulnerability_Median_2035'],
            mode='markers',
            marker=dict(
                size=10,
                color=df['Vulnerability_Median_2035'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="2035<br>Vulnerability"),
                line=dict(width=0.5, color='white')
            ),
            text=df['Occupation'],
            hovertemplate='<b>%{text}</b><br>' +
                         '2024 Vulnerability: %{x:.3f}<br>' +
                         '2035 Vulnerability: %{y:.3f}<br>' +
                         '<extra></extra>'
        )
    )
    
    # Add diagonal line (no change)
    fig3.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='gray', dash='dash', width=2),
            name='No Change Line',
            hoverinfo='skip'
        )
    )
    
    # Update scatter plot layout with fixed rangeselector
    fig3.update_layout(
        title={
            'text': '<b>Occupation Vulnerability: 2024 vs 2035</b><br>' +
                   '<sub>Points above diagonal line = increasing vulnerability | Use range selector to zoom</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis=dict(
            title='<b>2024 Vulnerability Score</b>',
            range=[0, 1],
            rangeslider=dict(visible=True, thickness=0.05),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='Reset', step='all'),
                    dict(count=1, label='High Risk (>0.7)', step='all'),
                    dict(count=1, label='Low Risk (<0.4)', step='all')
                ])
            )
        ),
        yaxis=dict(
            title='<b>2035 Vulnerability Score</b>',
            range=[0, 1]
        ),
        height=800,
        width=1200,
        template='plotly_white',
        hovermode='closest'
    )

    # Add update_xaxis and update_yaxis calls to handle range updates
    fig3.update_xaxes(range=[0, 1])
    fig3.update_yaxes(range=[0, 1])

    output_path_3 = CONFIG['output_html'].replace('.html', '_scatter.html')
    fig3.write_html(output_path_3)
    print(f"   ✅ Scatter plot explorer saved to {output_path_3}")
    
    # ============================================================
    # CREATE INTERACTIVE COMPARISON TOOL
    # ============================================================
    
    print("\n🎨 Creating occupation comparison tool...")
    
    # Get top 20 for dropdown options
    top_20 = df.nlargest(20, 'Vulnerability_Median_2035')
    
    # Create figure with multiple traces (one per occupation)
    fig4 = go.Figure()
    
    for idx, (_, occ) in enumerate(top_20.iterrows()):
        median_vals = [occ[f'Vulnerability_Median_{year}'] for year in CONFIG['years']]
        p05_vals = [occ[f'Vulnerability_p05_{year}'] for year in CONFIG['years']]
        p95_vals = [occ[f'Vulnerability_p95_{year}'] for year in CONFIG['years']]
        
        occ_name = occ['Occupation']
        
        # Add traces (initially all visible except first 3)
        visible = True if idx < 3 else 'legendonly'
        
        # P95 band upper
        fig4.add_trace(
            go.Scatter(
                x=CONFIG['years'],
                y=p95_vals,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                visible=visible,
                legendgroup=occ_name
            )
        )
        
        # P05 band lower with fill
        fig4.add_trace(
            go.Scatter(
                x=CONFIG['years'],
                y=p05_vals,
                mode='lines',
                line=dict(width=0),
                fillcolor=f'rgba({idx*12}, {100+idx*5}, {200-idx*8}, 0.2)',
                fill='tonexty',
                showlegend=False,
                hoverinfo='skip',
                visible=visible,
                legendgroup=occ_name
            )
        )
        
        # Median line
        fig4.add_trace(
            go.Scatter(
                x=CONFIG['years'],
                y=median_vals,
                mode='lines+markers',
                name=occ_name[:50],
                line=dict(width=3),
                marker=dict(size=8),
                visible=visible,
                legendgroup=occ_name,
                hovertemplate=f'<b>{occ_name}</b><br>Year: %{{x}}<br>Vulnerability: %{{y:.3f}}<extra></extra>'
            )
        )
    
    fig4.update_layout(
        title={
            'text': '<b>Compare Occupation Trajectories</b><br>' +
                   '<sub>Click legend items to show/hide occupations | Shaded areas show 90% confidence intervals</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='<b>Year</b>',
        yaxis_title='<b>Vulnerability Score</b>',
        height=700,
        width=1400,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        yaxis=dict(range=[0, 1])
    )
    
    output_path_4 = CONFIG['output_html'].replace('.html', '_comparison.html')
    fig4.write_html(output_path_4)
    print(f"   ✅ Comparison tool saved to {output_path_4}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print("\n" + "=" * 60)
    print("✅ DASHBOARD CREATION COMPLETE!")
    print("=" * 60)
    print("\nCreated 4 interactive dashboards:")
    print(f"  1. {CONFIG['output_html']}")
    print(f"     → Animated bar chart with year slider and play button")
    print(f"  2. {output_path_2}")
    print(f"     → Comprehensive multi-view analysis")
    print(f"  3. {output_path_3}")
    print(f"     → Scatter plot with range sliders")
    print(f"  4. {output_path_4}")
    print(f"     → Occupation comparison tool (click legend to toggle)")
    print("\n💡 Tips:")
    print("  • Use the year slider to explore temporal changes")
    print("  • Click 'Play' to animate through years")
    print("  • Click legend items to show/hide specific occupations")
    print("  • Hover over data points for detailed information")
    print("  • Use range sliders to zoom into specific regions")
    print("=" * 60)

if __name__ == "__main__":
    create_interactive_dashboard()