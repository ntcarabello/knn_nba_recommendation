import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
import dash_table as dt
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

year = 2019
url_per_100 = "https://www.basketball-reference.com/leagues/NBA_{}_per_poss.html".format(year)
html_per_100 = requests.get(url_per_100)
soup_per_100 = BeautifulSoup(html_per_100.text, 'html.parser')

url_adv = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html".format(year)
html_adv = requests.get(url_adv)
soup_adv = BeautifulSoup(html_adv.text, 'html.parser')

table_per_100 = soup_per_100.find_all('table')
table_adv = soup_adv.find_all('table')

nba_df_per_100 = pd.read_html(str(table_per_100))[0]
nba_df_per_100.drop(columns=['Unnamed: 29','Rk'],inplace=True)
nba_df_per_100 = nba_df_per_100.loc[~(nba_df_per_100['Player'] == 'Player')]
nba_df_per_100['G'] = nba_df_per_100['G'].astype(int)
nba_df_per_100['max_game_filter'] = nba_df_per_100.groupby(['Player'])['G'].transform('max')
nba_df_per_100 = nba_df_per_100.loc[nba_df_per_100['G'] == nba_df_per_100['max_game_filter']]
nba_df_per_100.drop(columns=['max_game_filter','Age','Tm','G','GS','MP'],inplace=True)


nba_df_adv = pd.read_html(str(table_adv))[0]
nba_df_adv.drop(columns=['Unnamed: 19','Unnamed: 24'],inplace=True)
nba_df_adv = nba_df_adv.loc[~(nba_df_adv['Rk'] == 'Rk')]
nba_df_adv['G'] = nba_df_adv['G'].astype(int)
nba_df_adv['max_game_filter'] = nba_df_adv.groupby(['Player'])['G'].transform('max')
nba_df_adv = nba_df_adv.loc[nba_df_adv['G'] == nba_df_adv['max_game_filter']]
nba_df_adv.drop(columns=['Rk','Pos','Age','Tm','G','MP','max_game_filter'],inplace=True)


nba_df = nba_df_per_100.merge(nba_df_adv, on='Player')

url_shooting = "https://www.basketball-reference.com/leagues/NBA_{}_shooting.html".format(year)
html_shooting = requests.get(url_shooting)
soup_shooting = BeautifulSoup(html_shooting.text,'html.parser')
table_shooting = soup_shooting.find_all('table')
nba_df_shooting = pd.read_html(str(table_shooting))[0]


nba_df_shooting.columns = ['|'.join(col).strip() for col in nba_df_shooting.columns.values]
nba_df_shooting['Player'] = nba_df_shooting['Unnamed: 1_level_0|Player'].str.lstrip('P')
nba_df_shooting = nba_df_shooting.loc[~(nba_df_shooting['Unnamed: 1_level_0|Player'] == 'Player')]


nba_df_shooting['Unnamed: 5_level_0|G'] = nba_df_shooting['Unnamed: 5_level_0|G'].astype(int)
nba_df_shooting['max_game_filter'] = nba_df_shooting.groupby(['Unnamed: 1_level_0|Player'])['Unnamed: 5_level_0|G'].transform('max')
nba_df_shooting = nba_df_shooting.loc[nba_df_shooting['Unnamed: 5_level_0|G'] == nba_df_shooting['max_game_filter']]
nba_df_shooting.drop(columns=['max_game_filter'],inplace=True)


nba_df_shooting = nba_df_shooting.loc[:,~nba_df_shooting.columns.str.startswith('Unnamed')]
nba_df_shooting = nba_df_shooting.loc[:,~(nba_df_shooting.columns.str.contains('Ast'))
                                      & ~(nba_df_shooting.columns.str.contains('Dunk'))
                                      & ~(nba_df_shooting.columns.str.contains('Heave'))]


nba_df = nba_df.merge(nba_df_shooting, on='Player')
nba_df.fillna(0,inplace=True)
nba_df_copy = nba_df.copy()
nba_df.drop(columns=['Pos'],inplace=True)
nba_df.set_index('Player',inplace=True)

nba_df_scaled = StandardScaler().fit_transform(nba_df)
nba_df_final = pd.DataFrame(index=nba_df.index, columns=nba_df.columns, data=nba_df_scaled)
nba_df_feature_matrix = csr_matrix(nba_df_final.values)

knn_search = NearestNeighbors(metric='cosine', algorithm='brute')
knn_search.fit(nba_df_feature_matrix)

player_list = []
rec_list = []

for player in nba_df_final.index:
    distances, indices = knn_search.kneighbors(nba_df_final.loc[player,:].values.reshape(1,-1), n_neighbors=11)

    for elem in range(0,len(distances.flatten())):
        if elem == 0:
            player_list.append([player])
        else:
            rec_list.append([player, elem, nba_df_final.index[indices.flatten()[elem]],distances.flatten()[elem]])

recommendation_df = pd.DataFrame(rec_list, columns=['Search Player','Recommendation Rank','Similar Player','Distance Score'])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
colors = {'text': 'white'}

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.layout = html.Div(style={'background-image': 'url(assets/nba-basketball-logo-nba.jpg)',
                             'background-position': 'center',
                             'position':'fixed',
                             'min-height':'100%',
                             # 'min-width':'100%',
                             'textAlign':'center',
                             'margin':'auto',
                             'width':'99%'

                }, children=[
    html.H1('NBA Player Recommendation Engine', style={'color': colors['text']}),
    dcc.Dropdown(
        id='nba_dropdown',
        options=[
            {'label': i, 'value': i}
            for i in nba_df_final.index
        ],
        placeholder="Select or type a player name from the 2018/19 NBA season for comparisons!",
        style = {'width': '100%',
                  'align-items': 'center',
                   'justify-content': 'center'
            }

    ),

    html.Div(id='output_container')
])

@app.callback(
    dash.dependencies.Output('output_container', 'children'),
    [dash.dependencies.Input('nba_dropdown', 'value')])

def update_output(player_value):
    recommendation_df_copy = recommendation_df[recommendation_df['Search Player']==player_value]
    data = recommendation_df_copy.to_dict('rows')
    columns =  [{'name': i, 'id': i,} for i in (recommendation_df_copy.columns)]
    if player_value is None:
        raise PreventUpdate
    else:
        return dt.DataTable(data=data,
                            columns=columns[1:],
                             style_cell={'textAlign': 'center',
                                        'fontSize':14
                                        },
                              style_as_list_view=True,
                              style_header={'backgroundColor': 'white',
                                            'fontWeight': 'bold'
                                            }
                            )

if __name__ == '__main__':
    app.run_server(debug=True)
