# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 10:38:16 2021

@author: YAHIYA
"""

colors = {
    'background': '#4444',
    'text': '#7FDBFF'
}

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
from datetime import date
import gunicorn
#import re

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

#---------------------------------------------------------------
##import data set
chunksize = 40000

df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv', chunksize=chunksize, iterator=True)

df = pd.concat(df, ignore_index=True)
pd.options.mode.chained_assignment=None
#df = pd.read_excel('https://covid.ourworldindata.org/data/owid-covid-data.xlsx')
#df = pd.read_csv('owid-covid-data.csv')
#PREPROCESS
### drop column that has 110000 (rows) null values
#df.columns[df.isna().sum()>110000]
df.drop(['weekly_icu_admissions', 'weekly_icu_admissions_per_million',
       'weekly_hosp_admissions', 'weekly_hosp_admissions_per_million',
       'total_boosters', 'total_boosters_per_hundred', 'excess_mortality'], axis = 1,inplace=True) 
df['date'] = pd.to_datetime(df.date)
df=df.loc[:,['date','location','new_tests','new_cases', 'new_deaths','total_cases', 'total_deaths','total_vaccinations','population_density','gdp_per_capita']]
#---------------------------------------------------------------
###2
#df2 = df.copy()
#df2=df.loc[:,:]
df2=df[['location', 'date','new_cases', 'new_deaths','total_cases', 'total_deaths']]
df2_ = ['location', 'date','new_cases', 'new_deaths','total_cases', 'total_deaths']
World = df2[df2.location=='World'].loc[:, df2_].set_index('date')
Sri_Lanka = df2[df2.location=='Sri Lanka'].loc[:, df2_].set_index('date')
RoW = (World.iloc[:, 1:] - Sri_Lanka.iloc[:, 1:]).replace(np.nan, 0)
RoW.insert(0, 'south', ['RoW']*len(RoW))
SL = df2.query("location=='Sri Lanka'")
def south(SL):
  
  if SL['location'] == 'Sri Lanka':
    return 'Sri Lanka'
#-------------------------

SL.loc[:,'south'] = SL.apply(south, axis=1)
def south(df2):
  
  if df2['location'] == 'Afghanistan':
    return 'SAARC'
  elif df2['location'] == 'Bangladesh':
    return 'SAARC'
  elif df2['location'] == 'Bhutan':
    return 'SAARC'
  elif df2['location'] == 'India':
    return 'SAARC'
  elif df2['location'] == 'Maldives':
    return 'SAARC'
  elif df2['location'] == 'Nepal':
    return 'SAARC'
  elif df2['location'] == 'Pakistan':
    return 'SAARC'
  elif df2['location'] == 'Sri Lanka':
    return 'SAARC' 
  elif df2['location'] == 'Asia':
    return 'Asia'
  else:
    return 'All countires without south asians'

df2.loc[:,'south'] = df2.apply(south, axis=1)
df2= df2.groupby(['date','south'])[['new_cases', 'new_deaths','total_cases', 'total_deaths']].sum().reset_index()
df2 =df2[['date','south','new_cases', 'new_deaths','total_cases', 'total_deaths']].set_index('date')
SL=SL[['date','south','new_cases', 'new_deaths','total_cases', 'total_deaths']].set_index('date')
Frame = [df2,SL,RoW]
df2 = pd.concat(Frame)


df2=pd.DataFrame(data=df2).reset_index()

###3
#df['date'] = pd.to_datetime(df['date'])
df['Test_to_detection'] = (df['new_tests']/df['new_cases'])
df['Test_to_detection'] = df['Test_to_detection'].replace(np.nan,0)
df3 = df.groupby(['date','location'], as_index=False)['Test_to_detection'].sum()
df3 = df3.set_index('date')
df3 = df3.loc['2020-01-01':'2022-12-31']
df3 = df3.groupby([pd.Grouper(freq="D"),'location'])['Test_to_detection'].sum().reset_index()

###4
#df4 = df.copy()
#df4 = df.loc[:,:]
df4=df[['date','location','new_cases', 'new_tests']]
df4 = df4.query("location=='Sri Lanka'")

###5
df5 = df.query("location == ['Finland', 'Denmark','Switzerland','Iceland','Norway','Netherlands','Sweden','New Zealand','Austria','Luxembourg','Australia','Israel','Ireland']")
df5.location.unique()
all_dims = ['date','total_vaccinations','total_deaths','population_density','gdp_per_capita']

#---------------------------------------------------------------



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = html.Div([
        html.Div(
                children=[
                        html.H1(children='NIBM Individual Dash Assignment', style={'textAlign': 'center'}),
                        html.H4(children='Dash: An advanced web application using Python in Dash.', 
                                style={'textAlign': 'center'}),
                ],
        style = {'color':'#FFD700', 'fontSize':25,"text-align": "center",'backgroundColor':'#121212'}
        ),
        html.Br(),
        html.Br(),
        html.Div([
        html.Div([#first graph
        html.Label(['Choose one feature:'],style={'font-weight': 'bold', "text-align": "left",'color':'gold'}),
        html.Div([dcc.Dropdown(
                        id="dropdown_1",
                        options=[{'label': 'total_cases', 'value': 'total_cases'},
                                 {'label':'new_cases','value': 'new_cases'},
                                 {'label':'new_deaths','value':'new_deaths'},
                                 {'label':'total_deaths','value':'total_deaths'}],
                                 value='total_cases',
                                 clearable=False,
                                 multi=False,
                                 style={'color':'blue','fontSize':20,'backgroundColor':'#5ADEFF'}),
                                        
                    ],style={'width': '50%'}),
            
        
        html.Label(['Choose date range:'],style={'font-weight': 'bold', "text-align": "left",'color':'gold'}),
        html.Div([dcc.DatePickerRange(
                    id='datepicker_1',
                    min_date_allowed=date(2020, 1, 22),
                    #max_date_allowed=date(2023, 9, 23),
                    initial_visible_month=date(2020, 5, 1),
                    start_date=date(2020,1,3),
                    end_date=date(2021, 6, 3)),
            ]), 
        html.Div([dcc.Graph(id="first_graph"),
                      ],style= {'height':'100%','width': '100%', 'display': 'inline-block'}),
             
            ]),
        html.Br(),
        html.Br(),
        html.Div([#second graph
                
        html.Label(['Choose one feature:'],style={'font-weight': 'bold', "text-align": "left",'color':'gold'}),
        html.Div([dcc.Dropdown(
                        id="dropdown_2_y",
                        options=[{'label': 'total_cases', 'value': 'total_cases'},
                                 {'label':'new_cases','value': 'new_cases'},
                                 {'label':'new_deaths','value':'new_deaths'},
                                 {'label':'total_deaths','value':'total_deaths'}],
                                 value='total_cases',
                                 clearable=False,
                                 multi=False,
                                 style={'color':'blue','fontSize':20,'backgroundColor':'#5ADEFF','border':'2px blue solid'}),
                
        html.Label(['Choose location :'],style={'font-weight': 'bold', "text-align": "left",'color':'gold'}),
        html.Div([dcc.Checklist(
                        id = 'checklist_2',
                        options=[
                                {"label": y, "value": y}for y in ['RoW', 'Asia', 'SAARC','Sri Lanka']],
                        value=['Sri Lanka'],
                        labelStyle={'backgroundColor':'#5ADEFF','border':'5px blue solid'},
                )],
                style={'width':'50%','border':'5px blue solid'}),
                
        html.Label(['Choose date range:'],style={'font-weight': 'bold', "text-align": "left",'color':'gold'}),
        html.Div([dcc.DatePickerRange(id='date_picker_range_2',
                                      min_date_allowed=date(2020, 1, 22),
                                      #max_date_allowed=date(2023, 9, 23),
                                      initial_visible_month=date(2020, 5, 1),
                                      start_date=date(2020,1,3),
                                      end_date=date(2021, 6, 3)),
                  ]),
                
        html.Label(['Choose aggregation method:'],style={'font-weight': 'bold', "text-align": "left",'color':'gold'}),
        html.Div([dcc.Dropdown(
                    id = 'dropdown_2_x',
                    options=[{'label': i, 'value': i} for i in ['Daily', 'Weekly Average', 'Monthly Average', '7-day average', '14-day average']],
                    value='Daily',
                    multi=False,
                    clearable=False,
                    style= {'color':'blue','fontSize':20,'backgroundColor':'#5ADEFF','border':'2px blue solid'}),#'backgroundColor':'#858275'
                            ]),
                ],style= {'width': '50%'}),
        
        html.Div([dcc.Graph(id='second_graph')],
                          style= {'height':'100%','width': '100%', 'display': 'inline-block'}),
                  ]),
        html.Br(),
        html.Br(),
        html.Div([#Third graph
            
        html.Label(['Choose Locations to Compare:'],style={'font-weight': 'bold', "text-align": "center",'color':'gold'}),
        html.Div([dcc.Dropdown(id='location_1',
                                       options=[{'label':x, 'value':x} for x in df3.sort_values('location')['location'].unique()],
                                       value='South Korea',
                                       multi=False,
                                       disabled=False,
                                       clearable=True,
                                       searchable=True,
                                       placeholder='Choose Location...',
                                       #style={'width':"90%"},
                                       persistence='string',
                                       persistence_type='memory',
                                       style={'color':'blue','fontSize':20,'backgroundColor':'#5ADEFF','border':'2px blue solid'}),
                                              ]), 
                
        
        html.Div([dcc.Dropdown(id='location_2',
                                   options=[{'label':x, 'value':x} for x in df3.sort_values('location')['location'].unique()],
                                   value='Sri Lanka',
                                   multi=False,
                                   clearable=False,
                                   persistence='string',
                                   persistence_type='local',
                                   style={'color':'blue','fontSize':20,'background-color':'#5ADEFF','border':'2px blue solid'})
                                          ]),
        
        html.Label(['Choose date range:'],style={'font-weight': 'bold', "text-align": "left",'color':'gold'}),
        html.Div([dcc.DatePickerRange(id='datepicker_3',
                                          min_date_allowed=date(2020, 1, 22),
                                          #max_date_allowed=date(2023, 9, 23),
                                          initial_visible_month=date(2020, 5, 1),
                                          start_date=date(2020,3,3),
                                          end_date=date(2021, 3, 8))]),
        html.Div([dcc.Graph(id='third_graph'),
                      ],style= {'height':'100%','width': '100%', 'display': 'inline-block'}),
                ]),
        html.Br(),
        html.Br(),
        html.Div([ #fourth graph
        html.Label(['Choose date range:'],style={'font-weight': 'bold', "text-align": "left",'color':'gold'}),
        html.Div([dcc.DatePickerRange(
                        id='datepicker_4',
                        min_date_allowed=date(2020, 1, 22),
                        #max_date_allowed=date(2023, 9, 23),
                        initial_visible_month=date(2021, 1, 5),
                        start_date=date(2020,1,3),
                        end_date=date(2021, 6, 3)),
        html.Div([dcc.Graph(id='fourth_graph')
                ],style= {'height':'100%','width': '100%', 'display': 'inline-block'}),
    
        
            ]),
            ],style={'color':'blue','fontSize':20,'backgroundColor':'#121212'}),
        html.Br(),
        html.Br(),
        html.Div([#fifth graph
                
        html.Label(['Choose features with date:'],style={'font-weight': 'bold', "text-align": "left",'color':'gold'}),
        html.Div([dcc.Dropdown(
                        id="dropdown_5",
                        options=[{"label": x, "value": x} 
                        for x in all_dims],
                        value=all_dims[:2],
                        multi=True,
                        style={'color':'blue','fontSize':20,'backgroundColor':'#5ADEFF','border':'2px blue solid'})]),
        
        html.Label(['**Choose Buttons with date']
                ,style={'font-weight': 'bold', "text-align": "left",'color':'gold',
                       'backgroundColor':'#121212','fontSize':12}),
                
        html.Div([dcc.Graph(id="fifth_graph"),
                          ], style= {'height':'100%','width': '100%', 'display': 'inline-block'}),
        
        html.Label(['Name :Sarah Zavahir (cohndds (f/t)201f-002)'],style={'font-weight': 'bold', "text-align": "center",'color':'silver','fontSize':20})
        ]),
        
            
    ]),
],style = {'backgroundColor':'#121212'})
#---------------------------------------------------------------

 
###1
@app.callback(
        Output("first_graph", "figure"),
        [Input('datepicker_1',"start_date")],
        [Input('datepicker_1',"end_date")],
        [Input("dropdown_1", "value")])


def update_graph(start_date, end_date, dropdown_1):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    fig_1 = px.line(df.query("location=='World'"), x = 'date',y = dropdown_1,template='plotly_dark')
    
    fig_1.update_layout(
        title={'text':'1. Time series analysis to the World',
                      'font':{'size':28},'x':0.5,'xanchor':'center'},
         
        xaxis_range=[start_date, end_date],
        font_family="Courier New",font_color="gold",title_font_family="Times New Roman",
        legend_title_font_color="orange")
    
    return fig_1

###2
@app.callback(
    Output('second_graph','figure'),
    Input('dropdown_2_y','value'),
    Input('checklist_2','value'),
    Input('date_picker_range_2',"start_date"),
    Input('date_picker_range_2',"end_date"),
    Input('dropdown_2_x','value'),
)

def update_fig(features, checklist_2,start_date, end_date, date_aggre):
    print(features, checklist_2,start_date, end_date, date_aggre)
    date_picker = df2[(df2.date>=start_date)&(df2.date<=end_date)] 
    
    
    checklist2_location = date_picker[date_picker.south.isin(checklist_2)]
    
    # filter by RoW, SAARK and Asia and aggregate by specified date
    if date_aggre=='Daily':
        title=f"2. {date_aggre} changes in {features}"
        df_2  = checklist2_location
    elif date_aggre=='Weekly Average':
        title=f"2. {date_aggre} changes in {features}"
        df_2 = checklist2_location.groupby([pd.Grouper(key='date',freq='W'), 'south'], ).mean().reset_index()
    elif date_aggre=='Monthly Average':
        title=f"2. {date_aggre} changes in {features}"
        df_2 = checklist2_location.groupby([pd.Grouper(key='date',freq='M'), 'south'], ).mean().reset_index()
    elif date_aggre=='7-day average':
        title=f"2. {date_aggre} changes in {features}"
        df_2 = checklist2_location.groupby('south').rolling(7, on='date').mean().reset_index()
    elif date_aggre=='14-day average':
        title=f"2. {date_aggre} changes in {features}"
        df_2=checklist2_location.groupby('south').rolling(14, on='date').mean().reset_index()
        
    
    fig_2 = px.line(df_2, x='date', y=features,color='south', title=title,template='plotly_dark')
    #fig_2.layout.title.font.size:28
    fig_2.layout.title.x = 0.5
    fig_2.update_layout(title={'font':{'size':28}},
                      xaxis_range=[start_date, end_date],
                      xaxis_title='Date',
                     yaxis_title=f'{features}',
                     font_family="Courier New",font_color="gold",
                     title_font_family="Times New Roman",
                     legend_title_font_color="orange")
    
    return fig_2

###3
@app.callback(
    Output('third_graph','figure'),
    [Input('location_1','value'),
    Input('location_2','value')],
    [Input('datepicker_3',"start_date")],
    [Input('datepicker_3', "end_date")]
)

def build_graph(first_loc,second_loc,start_date, end_date):
    dff=df3[(df3['location']==first_loc)|
           (df3['location']==second_loc)]
    
    

    fig_3 = px.line(dff, x="date", y="Test_to_detection",color='location',height=600,template='plotly_dark')# 
    fig_3.update_layout(yaxis={'title':'Test_to_detection Ratio'},
                      xaxis={'title':'Date'},
                      title={'text':'3. Test to detection Compared to Country',
                      'font':{'size':28},'x':0.5,'xanchor':'center'},
                       xaxis_range=[start_date, end_date],
                       font_family="Courier New",font_color="gold",title_font_family="Times New Roman",
                        legend_title_font_color="orange")
    #fig_3.update_xaxes(rangeslider_visible=True,template='plotly_white')#SLIDER
    return fig_3

###4
@app.callback(
    Output('fourth_graph','figure'),
    Input('datepicker_4','start_date'),
    Input('datepicker_4','end_date'))

def foruth_graph(start_date,end_date):
    
    mask = ((df4.date >= start_date)& (df4.date <= end_date))
    filtered_data = df4[mask]
    #filtered_data = df4.loc[[(df4.date >= start_date)& (df4.date <= end_date)],:]
    
    #filtered_data = df4.loc[[mask],["date","new_tests","new_cases"]].tolist()
    
    fig_4 = px.scatter(filtered_data, x="new_tests", y="new_cases",hover_data=['date'], 
                       trendline="ols",trendline_color_override='gold', template='plotly_dark')
    fig_4.update_layout(yaxis={'title':'New Cases'},
                      xaxis={'title':'New Tests'},
                      title={'text':'4. New Test vs New Cases with date ',
                      'font':{'size':28},'x':0.5,'xanchor':'center'},
                       font_family="Courier New",font_color="gold",title_font_family="Times New Roman",
                        legend_title_font_color="orange",
                        ) 
    fig_4.add_annotation(dict(font=dict(color='#d2cfc1',size=15),
                                        x=0,
                                        y=-0.12,
                                        showarrow=False,
                                        text="Up to date correlation is 0.63",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    return fig_4



###5
@app.callback(
    Output("fifth_graph", "figure"), 
    [Input("dropdown_5", "value")])
    

def update_bar_chart(dims):
    
   
    fig_5 = px.scatter_matrix(df5, dimensions=dims,color='location',template='plotly_dark',
                               )
    fig_5.update_layout(title={'text':'5.Total vaccination ,Total Deaths, GDP per Capita & Population Density compared to Happiest Countries 2020',
                               'font':{'size':24},'x':0.5,'xanchor':'center'},
                               font_family="Courier New",font_color="#FF11C2",
                        title_font_family="Times New Roman",
                        title_font_color='gold',
                        legend_title_font_color="orange")
    #fig_5.update_xaxes(rangeslider_visible=True)
    fig_5.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
    return fig_5
#---------------------------------------------------------------


if __name__ == '__main__':
    app.run_server(port = '5566')
    