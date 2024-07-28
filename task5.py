import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html

df = pd.read_csv('integrated_mobile_sales.csv')

df['Date'] = pd.to_datetime(df['Date'])


app = Dash(__name__)


bar_fig = px.bar(df.groupby('Brand')['TotalRevenue'].sum().reset_index(), 
                 x='Brand', y='TotalRevenue', title='Total Revenue by Brand')


line_fig = px.line(df.groupby('Date')['UnitsSold'].sum().reset_index(), 
                   x='Date', y='UnitsSold', title='Units Sold Over Time')


heatmap_data = df.pivot_table(index='Location', columns='PaymentMethod', values='CustomerSatisfaction', aggfunc='mean')
heatmap_fig = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='Viridis'))
heatmap_fig.update_layout(title='Customer Satisfaction by Location and Payment Method')


app.layout = html.Div(children=[
    html.H1(children='Mobile Sales Dashboard'),

    html.Div(children='''
        Interactive visualizations of mobile sales data.
    '''),

    dcc.Graph(
        id='bar-chart',
        figure=bar_fig
    ),

    dcc.Graph(
        id='line-chart',
        figure=line_fig
    ),

    dcc.Graph(
        id='heatmap',
        figure=heatmap_fig
    )
])


if __name__ == '__main__':
    app.run_server(debug=True)
