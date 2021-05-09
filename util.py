import plotly.graph_objects as go
import numpy as np


color_primary = 'rgb(0,34,80)'
color_secondary = 'rgb(185,145,43)'


# https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap
def plot_confusion_matrix(cm, labels, title):
    proportions = cm / np.sum(cm, axis=0).reshape((len(cm), 1))
    data = go.Heatmap(z=proportions, y=labels, x=labels, zmin=0, zmax=1,
                      colorscale=[(0, 'wheat'), (1, color_secondary)])
    annotations = []
    for j, row in enumerate(zip(cm, proportions)):
        for i, (value, p) in enumerate(zip(*row)):
            annotations.append({
                'x': labels[i],
                'y': labels[j],
                'font': {'color': 'black'},
                'text': str(value) + ' - {:.2%}'.format(p),
                'xref': 'x1',
                'yref': 'y1',
                'showarrow': False
            })
    layout = {
        'title': title,
        'xaxis': {'title': 'Predicted value'},
        'yaxis': {'title': 'Real value'},
        'annotations': annotations,
        'width': 800
    }
    fig = go.Figure(data=data, layout=layout)
    fig.show()
