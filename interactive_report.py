import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind, gaussian_kde
import os
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import webbrowser
from threading import Timer

# Create an output directory
output_dir = "/tmp/output"
os.makedirs(output_dir, exist_ok=True)

# Import the CSV file
data = pd.read_csv("all.csv")

# Function to perform t-test and generate statistics based on GDP percentile split
def analyze_split(data, percentile_threshold):
    """Analyze data based on a GDP percentile split"""
    data_sorted = data.sort_values(by='2021_GDP', ascending=False)
    
    # Calculate the cutoff index based on percentile
    cutoff_idx = int(len(data_sorted) * (percentile_threshold / 100))
    
    # Ensure we have at least one element in each group
    cutoff_idx = max(1, min(cutoff_idx, len(data_sorted) - 1))
    
    # Split the data
    upper_group = data_sorted.iloc[:cutoff_idx]
    lower_group = data_sorted.iloc[cutoff_idx:]
    
    results = {}
    
    # Analyze both groups
    for group, name in [
        (upper_group, f"Top {percentile_threshold}%"),
        (lower_group, f"Bottom {100-percentile_threshold}%")
    ]:
        # Perform t-test
        t_test_result = ttest_ind(group['2021_HE'], group['2019_HE'], nan_policy='omit')
        one_tailed_pvalue = t_test_result.pvalue / 2

        # Calculate descriptive statistics
        stats = {}
        for year in ['2019_HE', '2021_HE']:
            stats[year] = {
                "Mean": group[year].mean(),
                "Standard Error": group[year].std() / (len(group[year]) ** 0.5),
                "Median": group[year].median(),
                "Standard Deviation": group[year].std(),
                "Sample Variance": group[year].var(),
                "Kurtosis": group[year].kurtosis(),
                "Skewness": group[year].skew(),
                "Range": group[year].max() - group[year].min(),
                "Minimum": group[year].min(),
                "Maximum": group[year].max(),
                "Sum": group[year].sum(),
                "Count": group[year].count()
            }
        
        results[name] = {
            "data": group,
            "pvalue": one_tailed_pvalue,
            "statistic": t_test_result.statistic,
            "significant": one_tailed_pvalue < 0.05 and t_test_result.statistic > 0,
            "stats": stats
        }
    
    return results

def create_plots(percentile):
    """Create interactive plots based on the GDP percentile split"""
    # Get analysis results
    results = analyze_split(data, percentile)
    
    # Create subplot layout: 2 rows (for upper/lower groups) and 3 columns (for plot types)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            f"Boxplot ({list(results.keys())[0]})", 
            f"KDE Plot ({list(results.keys())[0]})", 
            f"Histogram ({list(results.keys())[0]})",
            f"Boxplot ({list(results.keys())[1]})", 
            f"KDE Plot ({list(results.keys())[1]})", 
            f"Histogram ({list(results.keys())[1]})"
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )
    
    # Colors for consistent styling
    colors = {'2019_HE': '#1E88E5', '2021_HE': '#FF8C00'}
    
    # Process each group
    for i, (group_name, result) in enumerate(results.items()):
        group = result['data']
        row = i + 1  # Row 1 for upper group, Row 2 for lower group
        
        # Create boxplot (column 1)
        for j, year in enumerate(['2019_HE', '2021_HE']):
            fig.add_trace(
                go.Box(
                    y=group[year].dropna(), 
                    name=year,
                    marker_color=colors[year],
                    boxmean=True  # Show mean as a dashed line
                ),
                row=row, col=1
            )
        
        # Create KDE plot (column 2)
        for j, year in enumerate(['2019_HE', '2021_HE']):
            # Calculate KDE data
            kde_data = group[year].dropna()
            if not kde_data.empty:
                kde_x = np.linspace(kde_data.min(), kde_data.max(), 1000)
                kde = gaussian_kde(kde_data)
                kde_y = kde(kde_x)
                
                fig.add_trace(
                    go.Scatter(
                        x=kde_x, 
                        y=kde_y,
                        mode='lines',
                        name=year,
                        fill='tozeroy',
                        line=dict(color=colors[year]),
                        opacity=0.6
                    ),
                    row=row, col=2
                )
        
        # Create histogram (column 3)
        for j, year in enumerate(['2019_HE', '2021_HE']):
            fig.add_trace(
                go.Histogram(
                    x=group[year].dropna(),
                    name=year,
                    marker_color=colors[year],
                    opacity=0.7,
                    nbinsx=15
                ),
                row=row, col=3
            )
            
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        title_text=f"GDP Split Analysis at {percentile}% cutoff",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        boxmode='group'
    )
    
    # Update axes labels
    for i in range(1, 3):  # Rows
        for j in range(1, 4):  # Columns
            fig.update_xaxes(title_text="Year" if j == 1 else "$ HEpp", row=i, col=j)
            fig.update_yaxes(title_text="$ HEpp" if j == 1 else "Density" if j == 2 else "Count", row=i, col=j)
    
    # Update histogram to overlay instead of stack
    fig.update_layout(barmode='overlay')
    
    return fig, results

# Create Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the app layout
app.layout = html.Div([
    html.H1("GDP Split Analysis", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.H3("Adjust GDP Percentile Split:", style={'display': 'inline-block', 'marginRight': '10px'}),
        html.Div(id='percentile-value', style={'display': 'inline-block', 'fontWeight': 'bold', 'fontSize': '18px'}),
        dcc.Slider(
            id='percentile-slider',
            min=5,
            max=95,
            step=5,
            value=50,
            marks={i: f'{i}%' for i in range(5, 96, 10)},
            updatemode='drag'
        ),
    ], style={'width': '80%', 'margin': '20px auto', 'textAlign': 'center'}),
    
    dcc.Graph(id='plot-area'),
    
    html.Div([
        html.H2("Hypothesis Test Results", style={'textAlign': 'center', 'color': '#2c3e50'}),
        
        html.Div([
            html.Div([
                html.H3(id='top-group-title'),
                html.Table([
                    html.Tr([
                        html.Th("Measure"),
                        html.Th("Result")
                    ]),
                    html.Tr([
                        html.Td("T-statistic"),
                        html.Td(id='top-t-stat')
                    ]),
                    html.Tr([
                        html.Td("One-tailed p-value"),
                        html.Td(id='top-p-value')
                    ]),
                    html.Tr([
                        html.Td("Significant"),
                        html.Td(id='top-significant')
                    ]),
                    html.Tr([
                        html.Td("Hypothesis (2021 > 2019)"),
                        html.Td(id='top-hypothesis')
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse', 'margin': '10px 0'}),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H3(id='bottom-group-title'),
                html.Table([
                    html.Tr([
                        html.Th("Measure"),
                        html.Th("Result")
                    ]),
                    html.Tr([
                        html.Td("T-statistic"),
                        html.Td(id='bottom-t-stat')
                    ]),
                    html.Tr([
                        html.Td("One-tailed p-value"),
                        html.Td(id='bottom-p-value')
                    ]),
                    html.Tr([
                        html.Td("Significant"),
                        html.Td(id='bottom-significant')
                    ]),
                    html.Tr([
                        html.Td("Hypothesis (2021 > 2019)"),
                        html.Td(id='bottom-hypothesis')
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse', 'margin': '10px 0'}),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'width': '100%', 'margin': '20px 0'}),
    ], style={'width': '90%', 'margin': '0 auto'}),
    
    html.Div([
        html.H2("Descriptive Statistics", style={'textAlign': 'center', 'color': '#2c3e50'}),
        
        html.Div([
            html.H3(id='top-stats-title'),
            html.Div(id='top-stats-table'),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.H3(id='bottom-stats-title'),
            html.Div(id='bottom-stats-table'),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'width': '90%', 'margin': '20px auto'})
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': '0 auto'})

# Add CSS for tables
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>GDP Split Analysis</title>
        {%favicon%}
        {%css%}
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
            }
            th {
                background-color: #f2f2f2;
                text-align: left;
            }
            td.numeric {
                text-align: right;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create callback for updating the displayed percentile
@app.callback(
    Output('percentile-value', 'children'),
    [Input('percentile-slider', 'value')]
)
def update_percentile_value(percentile):
    return f"{percentile}%"

# Create callback for updating everything based on percentile
@app.callback(
    [
        Output('plot-area', 'figure'),
        # Top group outputs
        Output('top-group-title', 'children'),
        Output('top-t-stat', 'children'),
        Output('top-p-value', 'children'),
        Output('top-significant', 'children'),
        Output('top-hypothesis', 'children'),
        Output('top-stats-title', 'children'),
        Output('top-stats-table', 'children'),
        # Bottom group outputs
        Output('bottom-group-title', 'children'),
        Output('bottom-t-stat', 'children'),
        Output('bottom-p-value', 'children'),
        Output('bottom-significant', 'children'),
        Output('bottom-hypothesis', 'children'),
        Output('bottom-stats-title', 'children'),
        Output('bottom-stats-table', 'children'),
    ],
    [Input('percentile-slider', 'value')]
)
def update_analysis(percentile):
    # Get plots and results
    fig, results = create_plots(percentile)
    
    # Get group names
    group_names = list(results.keys())
    top_group = group_names[0]
    bottom_group = group_names[1]
    
    # Create stats tables
    def create_stats_table(group_name):
        result = results[group_name]
        
        stat_names = {
            "Mean": "Mean",
            "Standard Error": "Std Err",
            "Median": "Median",
            "Standard Deviation": "Std Dev",
            "Sample Variance": "Variance",
            "Kurtosis": "Kurtosis",
            "Skewness": "Skewness",
            "Range": "Range",
            "Minimum": "Min",
            "Maximum": "Max",
            "Sum": "Sum",
            "Count": "Count"
        }
        
        table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Statistic"),
                    html.Th("2019 HE"),
                    html.Th("2021 HE")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(short_name),
                    html.Td(f"{result['stats']['2019_HE'][stat_name]:.2f}" if isinstance(result['stats']['2019_HE'][stat_name], float) else result['stats']['2019_HE'][stat_name], className='numeric'),
                    html.Td(f"{result['stats']['2021_HE'][stat_name]:.2f}" if isinstance(result['stats']['2021_HE'][stat_name], float) else result['stats']['2021_HE'][stat_name], className='numeric')
                ]) for stat_name, short_name in stat_names.items()
            ])
        ])
        
        return table
    
    # Return all updated components
    return (
        fig,
        # Top group
        top_group,
        f"{results[top_group]['statistic']:.4f}",
        f"{results[top_group]['pvalue']:.4f}",
        'Yes' if results[top_group]['significant'] else 'No',
        'True' if results[top_group]['significant'] else 'False',
        f"Statistics for {top_group}",
        create_stats_table(top_group),
        # Bottom group
        bottom_group,
        f"{results[bottom_group]['statistic']:.4f}",
        f"{results[bottom_group]['pvalue']:.4f}",
        'Yes' if results[bottom_group]['significant'] else 'No',
        'True' if results[bottom_group]['significant'] else 'False',
        f"Statistics for {bottom_group}",
        create_stats_table(bottom_group)
    )

def open_browser():
    """Open browser to the app URL"""
    webbrowser.open_new("http://127.0.0.1:8050/")

def run_dash_app():
    """Run the Dash app"""
    # Open browser after a short delay
    Timer(1, open_browser).start()
    
    # Start the Dash app - updated from run_server to run
    app.run(debug=False, port=8050)

# Main function to run the analysis
def run_analysis():
    """Main function to run the interactive dashboard"""
    print("Starting GDP Split Analysis Dashboard...")
    print("Opening web browser to: http://127.0.0.1:8050/")
    run_dash_app()

# Run the analysis when the script is executed
if __name__ == "__main__":
    run_analysis()
