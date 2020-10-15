import json
import math

from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly

from .utils import load_dataframe, load_pickle, tokenize


app = Flask(__name__)

df = load_dataframe()
model = load_pickle()


@app.route('/')
@app.route('/index')
def index():
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = list(df.columns[4:])
    category_counts = df[category_names].sum().values

    graphs = [
        {
            'data': [
                plotly.graph_objs.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'xaxis': {'title': 'Genre'},
                'yaxis': {'title': 'Count'}
            }
        },
        {
            'data': [
                plotly.graph_objs.Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'xaxis': {'title': 'Category'},
                'yaxis': {'title': 'Count'}
            }
        }
    ]

    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graph_json=graph_json)


@app.route('/go')
def go():
    query = request.args.get('query', '')

    class_columns = [column.replace('_', ' ').title() for column in df.columns[4:].values]
    class_labels = model.predict([query])[0]
    class_summary = ', '.join([class_columns[i] for i in range(len(class_columns)) if class_labels[i]])

    data = []
    number_cols = 4

    for i in range(math.ceil(len(class_columns) / number_cols)):
        row = []
        for j in range(number_cols):
            k = (i * number_cols) + j
            if k < len(class_columns):
                row.append((class_columns[k], class_labels[k]))
            else:
                row.append(('', 0))
        data.append(row)

    return render_template(
        'go.html',
        query=query,
        summary=class_summary,
        table=data
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()