
# import modules
import numpy as np
import dice_ml
from dice_ml.utils import helpers  # helper functions from DiCE ML to import pre-processed dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import random
from metrics import *
from random import randrange
import math 

import dash
from dash import html, dcc, Output, Input, dash_table, State
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.express as px


import math

# load income dataset from dice_ml helpers
dataset = helpers.load_adult_income_dataset()
# description of transformed features
adult_info = helpers.get_adult_data_info()
dataset.rename(columns={"gender": "sex"}, inplace=True)

# Random Seed at file level
random_seed = 42

np.random.seed(random_seed)
random.seed(random_seed) # just in case they are different 

def prepare_data(dataset, target_name, test_size, random_state):
    target =  dataset[target_name]
    train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                    target,
                                                                    test_size=test_size,
                                                                    random_state=random_state,
                                                                    stratify=target)
    x_train = train_dataset.drop(target_name, axis=1)
    x_test = test_dataset.drop(target_name, axis=1)
    return train_dataset, test_dataset, x_train, y_train, x_test, y_test

def prepare_pipeline(numerical, categorical, continuous_scaler = None):
    categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    if continuous_scaler:
        continuous_transformer = Pipeline(steps=[
            ('scaler', continuous_scaler)])
    
        transformations = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical),
                ('cont',continuous_transformer, numerical)])
    else:
        transformations = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical)], remainder='passthrough')

    clf = Pipeline(steps=[('preprocessor', transformations),
                        ('classifier', RandomForestClassifier())])
    
    return clf

# prepare data:
target_name = "income"
train_dataset, test_dataset, x_train, y_train, x_test, y_test = prepare_data(
    dataset, 
    target_name, 
    0.2,
    0)

# prepare pipeline:
numerical_names = ["age", "hours_per_week"]
categorical_names = x_train.columns.difference(numerical_names)
clf = prepare_pipeline(numerical_names, categorical_names)

# fit model:
model = clf.fit(x_train, y_train)

# Prepare Dice data, model, and exp generator:
d = dice_ml.Data(dataframe=train_dataset, continuous_features=numerical_names, outcome_name=target_name)

# Use sklearn backend
m = dice_ml.Model(model=model, backend="sklearn")

# Use method=random for generating CFs, may try genetic method as well later
exp = dice_ml.Dice(d, m, method="random")

# choose a query
query_index = 0
query = x_test.iloc[[query_index]]

# identify actionable features
actionable_features_np = np.array([1, 1, 1, 0, 1, 0, 0, 1])
actionable_features = query.columns[actionable_features_np]
mad_dict = mad(x_train, numerical_names) 

# specify model information
model_info = {
       'model' : model,
       'training_data': x_train,
       'testing_data': x_test,
       'training_labels': y_train,
       'testing_labels': y_test,
       'scaler': clf['preprocessor'],
       'dist_function': dist, # mixed dist from metrics_new
       'actionable_features_np': actionable_features_np,
       'mad': mad_dict,
}

# target is opposit class
target = 'opposite'

# create dataframe to store results and other info
fields = ['model type', 
          'CF method','test sample', 'k', 
          'desired range', 'features to vary', 'counterfactuals', 'preds',
          'execution time','validity', 'sparsity', 'diversity', 
          'feasibility', 'actionability', 'proximity', 'stability'  ]

results = pd.DataFrame({}, columns=fields)

# For each configuration, run twice for the sake of measuring stability. 
numruns = 2

cont_features = query.select_dtypes(include='number').columns

def experiment_run (model_info, numruns, query, target_range, k, actionable_features, results, exp_model=exp):
      # print("query is",query)
      cf_stab_list = []

      opp = 1.0 - clf.predict(query)[0]
      # print("opp is ", opp)

      if target_range == 'opposite':
            for num in list(range(numruns)):
                  # print(f'on run {num}')
                  cf_stab = exp_model.generate_counterfactuals(query, total_CFs=k, desired_class=target_range)

                  # print("cfs returns ",cf_stab)


                  # cf_stab.visualize_as_dataframe(show_only_changes=True,display_sparse_df=False)
                  # # print(cf_stab)
                  cf_as_df =  cf_stab.cf_examples_list[0].final_cfs_df.reset_index()
                  cf_preds = cf_stab.cf_examples_list[0].final_cfs_df.values[:,-1]
                  cf_as_array = cf_stab.cf_examples_list[0].final_cfs_df.values[:,:-1]

                  # # print('array', cf_as_array)
                  # # print('df', cf_as_df)
                  
                  # append to list for stability calculation
                  cf_stab_list.append(cf_as_array)
                  
                  # calculate scores
                  cf_output = model_info['model'].predict(cf_as_df)
                  score_validity = val(cf_output, opp, k)
                  
                  # score_feasibility = 0
                  score_feasibility = impl(
                        query, 
                        cf_as_df, 
                        model_info["training_data"], 
                        model_info["scaler"], 
                        model_info["dist_function"],
                        model_info["mad"],
                        cont=cont_features
                        )
                                    
                  score_actionability = act(
                        query.values, 
                        cf_as_array, 
                        model_info["actionable_features_np"]
                        )
                  
                  score_proximity = prox(
                        query, 
                        cf_as_df,  
                        model_info["scaler"],  
                        model_info["dist_function"],
                        model_info["mad"],
                        cont=cont_features
                        )

                  score_sparsity = spar(
                        query.values, 
                        cf_as_array, 
                        dp=1
                        )

                  score_diversity = div_count(cf_as_array)
                  
                  # log all scores in results dataframe
                  results.loc[len(results)] = [
                        "classification", 
                        "DiCE random", 
                        query, 
                        k, 
                        target_range, 
                        actionable_features, 
                        cf_as_array, 
                        cf_preds,
                        0, 
                        score_validity, 
                        score_sparsity, 
                        score_diversity, 
                        score_feasibility, 
                        score_actionability, 
                        score_proximity, 
                        0]

      score_stability = stab(cf_stab_list[0], cf_stab_list[1])
      
      # update stability score on every second example!
      results.at[len(results)-1, 'stability'] = score_stability

      return results

# choose a query
query_index = 21
query = x_test.iloc[[query_index]]
# run experiment RANDOM for k = 1 up to 5
results=pd.DataFrame({}, columns=fields)

for k in list(range(1,5)):
    results = experiment_run(model_info, numruns, query, target, k, actionable_features, results=results)

def plot_radar(x, num_cf):
    df = pd.DataFrame(dict(
    r=x,
    # theta=['Spar ⬆️ (Quant)', ' Impl ⬇️ (Qual)', 'Val ⬆️ (Rel)', 'Prox ⬇️ (Mann)',  ],
    theta=[' Quant (Spar)', ' Qual (Feas)', 'Rel (Val)', ' Mann (Prox)',  ],
    ))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, title=f"CF #{num_cf}")
    fig.update_traces(fill='toself')

    return fig

fig = plot_radar([0.5, 0.5, 1.0, 0.8], 1)

def choose_query(query_index = 0):
    # choose a query
    
    query = x_test.iloc[[query_index]]

    # run experiment for k = 1 up to 5
    results=pd.DataFrame({}, columns=fields)
    for k in list(range(1,5)):
        results = experiment_run(model_info, numruns, query, target, k, actionable_features, results=results)
    return query_index, query, results

query_index, query, results = choose_query(0)

def feature_diff(query, cf, cols):
    return cols[~np.equal(query, cf)[0]].values

query = query
numruns = 2
opp = 1.0 - model_info['model'].predict(query)

# fig = plot_results(results[1::2])
radar = plot_radar([0,0,0,0], 1)

df = pd.DataFrame(np.array((results[1::2].loc[results['k'] == 1]["counterfactuals"].values[0])), columns=query.columns.values)
df["outcome"] = np.array((results[1::2].loc[results['k'] == 1]["preds"].values[0]))
df["id"] = df.index
# print(df)

def feature_diff(query, cf, cols):
    return cols[~np.equal(query, cf
    )[0]].values

card_content = [
    dbc.CardHeader("Card header"),
    dbc.CardBody(
        [
            html.H5("Card title", className="card-title"),
            html.P(
                "This is some card content that we'll reuse",
                className="card-text",
            ),
        ]
    ),
]

query_content = [
    dbc.CardHeader("Query"),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Button("🎲", color="dark", id="rnd"), width="auto"),
                    dbc.Col(html.H5(f"Original outcome: {y_test.iloc[[query_index]].values}", className="card-title"),)
                    
                ],
                className="mb-4",
            ),
        
            dbc.Spinner(
                [
                    dcc.Store(id="store"),
                    html.Div([
                dash_table.DataTable(
                    style_table={'overflowX': 'scroll'},
                    data=query.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in query.columns if i != "id"],
                    id='query')
                    ])
                ],
            ),
            
            
        ]
    ),
]

cf_content = [

    dbc.CardHeader("Response"),
    dbc.CardBody(
        html.Div(
            [
                html.Div("Select number of responses (k)", id="pagination-contents"),
                dbc.Pagination(id ="pagination", max_value=len(results[1::2]), active_page=1),
                html.H5("Counterfactual(s) generated with random restart algorithm:", className="card-title"),

                            html.Div([
                                dbc.Col(html.Div([
                                    
                                    # html.H3("Counterfactual Instances"),
                                    # dbc.Table.from_dataframe(query, striped=True, bordered=True, hover=True, id = "counterfactual-df"),
                                    html.Div([dash_table.DataTable(
                                        style_table={'overflowX': 'scroll'},
                                        row_selectable="multi",
                                        data=df.to_dict('records'),
                                        columns=[{"name": i, "id": i} for i in df.columns if i != "id"],
                                        id='counterfactual-df')]),
                                    html.Span("select an example", id="selected-example", style={"verticalAlign": "middle"}),]), width="auto"),
                                    ])
            
            ]
)
                
    ),
]

row_1 = dbc.Row(
    [
        dbc.Col(dbc.Card(query_content, color="success", outline=True))
        
    ],
    className="mb-4",
)
row_2 = dbc.Row(

    [
        
        dbc.Col(dbc.Card(cf_content, color="primary", outline=True))
    ],
    className="mb-4",
)

row_x = dbc.Row(
    [
        dbc.Col(dbc.Card(card_content, color="primary", outline=True)),
        dbc.Col(dbc.Card(card_content, color="warning", outline=True)),
        dbc.Col(dbc.Card(card_content, color="danger", outline=True)),
    ],
    className="mb-4",
)

row_3 = dbc.Row(

    [
        dbc.Spinner(
            [
                dcc.Store(id="store"),
                html.Div(id="tab-content", className="p-4"),
            ],
        ),
    ]

)

cards = html.Div([row_1, row_2, row_3])

app = dash.Dash(external_stylesheets=[dbc.themes.LITERA], suppress_callback_exceptions=True)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "marginLeft": "18rem",
    "marginRight": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("     💎", className="display-4"),
        html.H2("PRISM", className="display-4"),
        html.Hr(),
        html.P(
            "Explore counterfactual examples through a Gricean lens.", className="lead"
        ),
        dbc.Nav(
            [
                # dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("ADULT", href="/", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return cards
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

@app.callback(
    [Output("query", "data"),
     Output('counterfactual-df','data')], 
    [Input("rnd", "n_clicks")],
    prevent_initial_call=True,
)
def on_button_click(n):
    # print("in button click")
    global query, results
    query_index, query, results = choose_query(randrange(len(x_test)))
    df = pd.DataFrame(np.array((results[1::2].loc[results['k'] == 1]["counterfactuals"].values[0])), columns=query.columns.values)
    df["outcome"] = np.array((results[1::2].loc[results['k'] == 1]["preds"].values[0]))
    data = df.to_dict('records')
    if n is None:
        return dash.no_update
    else:
        # print("button clicked", n)
        return query.to_dict('records'), df.to_dict('records')

@app.callback(
    Output("tab-content", "children"),
    [Input("store", "data")],
)
def render_tab_content(data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """

    global fig, radar

    cols = []
    
    if data is not None:
        # # print("data is", data)

        for data_plot, diff in zip(data["scatter"], data["diffs"]):
            radar_content = [
                # dbc.CardHeader("PRISM Score"),
                dbc.CardHeader(f"Changes: {diff}"),
                dbc.CardBody(
                    [
                        # html.H5(f"CF #{cf}", className="card-title"),
                        # html.P(
                        #     "This is some card content that we'll reuse",
                        #     className="card-text",
                        # ),
                        html.Div(
                            [
                                dcc.Graph(figure = data_plot)
                            ]
                        )
                    ]
                ),
            ]

            cols.append(dbc.Col(dbc.Card(radar_content, color="light", outline=True), width=6))
        
        
        
        n_cards = len(cols)

        return dbc.Row( cols )

    return "No counterfactual example selected"


@app.callback(
    [Output("pagination-contents", "children"),
    Output("counterfactual-df", "data", allow_duplicate=True),
    Output("counterfactual-df", "selected_rows"),
    # Output("gen-counterfactual-df", "data", allow_duplicate=True),
    # Output("gen-counterfactual-df", "selected_rows"),
    ],
    [Input("pagination", "active_page")],
    prevent_initial_call=True,
)
def change_page(page):
    global results, results_gen
    df = pd.DataFrame(np.array((results[1::2].loc[results['k'] == 1]["counterfactuals"].values[0])), columns=query.columns.values)
    data = df.to_dict('records')
    if page:
        df = pd.DataFrame(np.array((results[1::2].loc[results['k'] == page]["counterfactuals"].values[0])), columns=query.columns.values)
        # # print(df)
        df["outcome"] = np.array((results[1::2].loc[results['k'] == page]["preds"].values[0]))
        data = df.to_dict('records')
        
        # gen_df = pd.DataFrame(np.array((results_gen[1::2].loc[results_gen['k'] == page]["counterfactuals"].values[0])), columns=query.columns.values)
        # # # print(df)
        # gen_df["outcome"] = np.array((results_gen[1::2].loc[results_gen['k'] == page]["preds"].values[0]))
        # gen_data = gen_df.to_dict('records')
        return f"k selected: {page}", data, []
    return "Select a value for k", data, []

@app.callback(
    [
        Output("selected-example", "children"), 
        Output("store", "data", allow_duplicate=True),

     ],
    Input("counterfactual-df", "selected_rows"),
    State("pagination", "active_page"),
    prevent_initial_call=True,
)
def style_selected_rows(sel_rows, k):
    global results, query, model_info, radar, fig

    eor_results = results[1::2]
    if len(sel_rows)==0:                                                                                                                                                                                                                      
        return dash.no_update
  
    # print(k)
    # print(sel_rows)
    selected_cf = eor_results.loc[eor_results['k'] == k]
    plots = []
    diffs = []
    for sel_row in sel_rows:
        # selected_cf['counterfactuals'].values[0][sel_row]
        # # print("selected row is", sel_row)

        # selected_cf = eor_results.loc[eor_results['k'] == k]
        cf_as_array = selected_cf['counterfactuals'].values[0][sel_row]
        cf_as_array = np.expand_dims(cf_as_array, axis=0)
        cf_as_df = pd.DataFrame(cf_as_array, columns=query.columns)

        diff = feature_diff(query.values, cf_as_array, query.columns)
        # # print('diffs are', diffs)
        diffs.append(diff)

        # calculate scores
        cf_output = model_info['model'].predict(cf_as_df)
        opp = 1.0 - model_info['model'].predict(query)
        score_validity = val(cf_output, opp, 1)

        score_feasibility = impl(
            query, 
            cf_as_df, 
            model_info["training_data"], 
            model_info["scaler"], 
            model_info["dist_function"],
            model_info["mad"],
            cont= query.select_dtypes(include='number').columns,
            cat = query.select_dtypes(include='object').columns
            )
                        
        score_actionability = act(
            query.values, 
            cf_as_array, 
            model_info["actionable_features_np"]
            )
        
        score_proximity = prox(
            query, 
            cf_as_df,  
            model_info["scaler"],  
            model_info["dist_function"],
            model_info["mad"],
            cont= query.select_dtypes(include='number').columns,
            cat = query.select_dtypes(include='object').columns
            )

        score_sparsity = spar(
            query.values,  
            cf_as_array, 
            dp=1
            )

        score_diversity = div_count(cf_as_array)
        # radar = plot_radar([score_sparsity, math.exp(-score_feasibility), score_validity, math.exp(-score_proximity)],sel_row+1)
        radar = plot_radar([math.exp(-(1-score_sparsity)), math.exp(-score_feasibility) , math.exp(-(1-score_validity)), math.exp(-score_proximity)],sel_row+1)
        plots.append(radar)
    
    # return f"validity: {score_validity}, sparsity: {score_sparsity}, feasibility: {score_feasibility}, proximity: {score_proximity} ",  {"scatter": radar, "all": fig}
    return f" ",  {"scatter": plots, "all": fig, "diffs": diffs}

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8000)
