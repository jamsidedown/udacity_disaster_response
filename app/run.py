import json

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

    columns = [column.replace('_', ' ').title() for column in df.columns[4:].values]

    print(columns)

    classification_labels = model.predict([query])[0]

    print(classification_labels)

    classification_results = dict(zip(columns, classification_labels))

    print(classification_results.items())

    classification_summary = ', '.join([key for key, value in classification_results.items() if value])

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        classification_summary=classification_summary
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()