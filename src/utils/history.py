import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

# pio.renderers.default = 'png'
pio.templates.default = 'plotly_white'


class History:
    def __init__(self, history_path: str | Path) -> None:
        """
        Initializes the History class by loading the training metrics from the specified history path.

        Args:
            history_path (str): The path to the directory containing the training metrics.
        """
        self.df = pd.read_csv(os.path.join(history_path, 'metrics.csv'))
        self.df.drop(columns=['step', 'epoch'], inplace=True)
        df = {col: self.df[col].dropna().tolist() for col in self.df.columns}
        self.df = pd.DataFrame(df)
        self.output_path = os.path.join(history_path, 'plots')
        self.epochs = self.df.index.to_series() + 1
        self.layout_config = dict(
            font=dict(size=16),
            width=1200,
            height=600,
            title_x=0.5,
            xaxis=dict(tickmode='linear', tick0=1, dtick=10),
        )
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
    def save_metric(self, column_name: str, metric_name: str):
        """
        Saves a specified metric from the training history as a png image.

        Args:
            column_name (str): The name of the column in the history dataframe.
            metric_name (str): The name of the metric to be saved.
        """
        metrics_fig = go.Figure()
        # Train metric
        metrics_fig.add_trace(
            go.Scatter(
                x=self.epochs,
                y=self.df[f'train_{column_name}'],
                name=f'train_{metric_name}',
                marker=dict(color='#00BFFF')
            )
        )
        # Val metric
        metrics_fig.add_trace(
            go.Scatter(
                x=self.epochs,
                y=self.df[f'val_{column_name}'],
                name=f'val_{metric_name}',
                marker=dict(color='#DC143C')
            )
        )
        metrics_fig.update_xaxes(title='<b>Epoch</b>')
        metrics_fig.update_yaxes(title=f'<b>{metric_name}</b>')
        metrics_fig.update_layout(title='<b>History</b>', overwrite=False, **self.layout_config)
        # Save
        output_path = os.path.join(self.output_path, f'{metric_name}.png')
        metrics_fig.write_image(output_path)
        return metrics_fig
    def save_lr(self, column_name: str):
        """
        Saves the learning rate history as a png image.

        Args:
            column_name (str): The name of the column in the history dataframe.
        """
        lr_fig = go.Figure()
        lr_fig.add_trace(
            go.Scatter(
                x=self.epochs,
                y=self.df[column_name],
                marker=dict(color='#a653ec')
            )
        )
        lr_fig.update_xaxes(title='<b>Epoch</b>')
        lr_fig.update_yaxes(title='<b>Learning rate</b>')
        lr_fig.update_layout(
            title='<b>Learning rate history</b>',
            overwrite=False,
            **self.layout_config,
            yaxis=dict(exponentformat='power')
        )
        # Save
        output_path = os.path.join(self.output_path, 'lr.png')
        lr_fig.write_image(output_path)
        return lr_fig