import pandas as pd
import os
import platform

# from DNN_plotly_functions_v001 import visualize_urd
# from DNN_h2o_functions_v001 import create_h2o_urd_model, get_predictions
from DNN_highcharts_functions_v001 import make_highcharts, visualize_urd_highcharts
import pandas as pd
import h2o
from h2o import exceptions
from h2o.automl import H2OAutoML
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators import H2ODeepWaterEstimator
import platform
import sys

import time

##### Start h2o#######
#h2o.init(nthreads=71, max_mem_size='30G')
#h2o.init(ip="192.168.0.11",strict_version_check=False)

# Remove all objects from h2o
#h2o.remove_all()

# Start H2O

###################################

# Define home directory
home_path = None
if platform.system() == 'Linux':
    home_path = os.path.expanduser("~")
elif platform.system() == 'Windows':
    home_path = 'C:\\'

# Import URD data
urd_path = os.path.join(home_path, '0MyDataBases/40Python/J_inertia/inertia.csv')
data_full = pd.read_csv(urd_path)
data_full['Time']=pd.to_datetime(data_full['Time'])

# Define list of pandas DataFrames for model to predict on
base_data_path = os.path.join(home_path, '0MyDataBases/40Python/J_inertia')
l_csv_test_data = []  # 'ExportFileWeather_2015.csv', 'ExportFileWeather_2014.csv', 'ExportFileWeather_2013.csv',
# 'ExportFileWeather_2012.csv', 'ExportFileWeather_2011.csv', 'ExportFileWeather_2010.csv']



# Set start and end dates for training
date_start = '1/1/2013 0:00'
date_end = '1/1/2015 0:00'

# Find row indices of training data
start_row = data_full[data_full['Time'] == date_start].index.tolist()[0]
end_row = data_full[data_full['Time'] == date_end].index.tolist()[0]

# Create training data slice and convert to training and validation H2OFrames

############################################################################333
## Starting h2o part
h2o.init(ip='localhost', strict_version_check=False)


pd_train = data_full[start_row:end_row].copy()
pd_train['Time']=pd.to_datetime(pd_train['Time'])

pd_test = data_full[end_row:].copy()
pd_test['Time']=pd.to_datetime(pd_test['Time'])


train = h2o.H2OFrame(pd_train,
                     column_types=['time', 'real', 'real', 'real', 'enum'],
                     destination_frame='Training_Validation_Frame')
training, validation = train.split_frame(ratios=[0.8])

test=h2o.H2OFrame(pd_test,
                     column_types=['time', 'real', 'real', 'real', 'enum'],
                     destination_frame='Test_Frame')
# Define predictors and response
predictors = ['Time','Wind','Load','if_special']
response = 'Inertia'

# Run DNN
model = H2ODeepLearningEstimator(model_id='inertia_first_try', epochs=5000, hidden=[100,100,100], activation="Rectifier",
                                     l1=0, l2=0,  stopping_metric='MSE')

model.train(x=predictors, y=response, training_frame=training, validation_frame=validation)

# Save DNN model to /tmp folder
# h2o.save_model(urd_model, path=save_path, force=True)

perf=model.model_performance(test_data=test)

# Get model predictions on test data, put directly in pandas DataFrames inside dictionary
pred = model.predict(test_data=test)



predict=pd_test.copy()
predict['predict']=pred.as_data_frame()['predict']
##################################################3

##############################################
#### Automl


# Run AutoML for 30 seconds
aml = H2OAutoML(max_runtime_secs = 1800)
aml.train(x=predictors, y=response, training_frame= train, validation_frame=validation, leaderboard_frame = test)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb
pred_aml=aml.predict(test)
#### testing ################################################################

predict=pd_test.copy()
predict['predict_dnn']=pred.as_data_frame()['predict']
predict['predict_aml']=pred_aml.as_data_frame()['predict']




# plotting plotly


import plotly.plotly as py
import plotly.graph_objs as go
import plotly
# Create random data with numpy
import numpy as np

N = 500
random_x = np.linspace(0, 1, N)
random_y = np.random.randn(N)

trace = go.Scatter(
    x = predict['Time'],
    y = predict['predict']
)


trace2 = go.Scatter(
    x = predict['Time'],
    y = predict['Inertia']
)

data = [trace2]

# Create layout for plotly
layout = dict(
    title='Predictions',
    xaxis1=dict(title='', rangeslider=dict(thickness=0.015, borderwidth=1), type='date', showgrid=True),
    # updatemenus=list([
    #     dict(
    #         buttons=[vis_dict for vis_dict in l_vis_dicts],
    #         type='buttons',
    #         active=0,
    #     )])
            )

# layout.update(d_axis_domains)  # Update layout with yaxis# keys

# Plot with plotly
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='bla')

# py.iplot(data, filename='basic-line')




# import plotly.plotly as py
# import plotly.graph_objs as go
#
# # Create random data with numpy
# import numpy as np
#
# # Create a trace
# trace = go.Scatter(
#     x = predict['Time'],
#     y = predict['predict']
# )
#
# data = [trace]
#
# py.iplot(data, filename='basic-line')
#



##################################################3
# plotting
# from bokeh.models import ColumnDataSource
# from bokeh.models import BoxAnnotation
# from bokeh.plotting import figure, show, output_file
# # from bokeh.sampledata.glucose import data
#
# TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
#
# # reduce data size
# # data = data_full #.ix['2010-10-06':'2010-10-13']
#
# del p
# # p=figure( tools=TOOLS, title="Inertia")
# p = figure(x_axis_type="datetime", plot_width=1200, plot_height=800, tools=TOOLS, title="Inertia")
# p.xgrid.grid_line_color=None
# p.ygrid.grid_line_alpha=0.5
# p.xaxis.axis_label = 'Time'
# p.yaxis.axis_label = 'Value'
#
#
# predict=pd_test.copy()
# predict['predict']=pred.as_data_frame()['predict']
# # p.line(data_full['Time'],data_full['Inertia'])
#
# p.line(predict['Time'],predict['predict'])
# # p.line(pd_test['Time'],pd_test['Inertia'])
#
# # p.line(pd_test['Time'],list(predict.values))
#
# # ds=ColumnDataSource(data_full)
# # p.line(source=ds, x='Time', y='Inertia')
# output_file("box_annotation1.html", title="box_annotation.py example")
#
# show(p)


#
#
# chart_path=os.path.join(home_path, '0MyDataBases/40Python/J_inertia/charts')
# # Plot with highcharts using external .py file function
# visualize_urd_highcharts(data_full, pd_test, chart_path)

