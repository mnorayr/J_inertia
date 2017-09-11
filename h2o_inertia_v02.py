import pandas as pd
import os
import platform

# from DNN_plotly_functions_v001 import visualize_urd
# from DNN_h2o_functions_v001 import create_h2o_urd_model, get_predictions
# from DNN_highcharts_functions_v001 import make_highcharts, visualize_urd_highcharts
import pandas as pd
import h2o
from h2o import exceptions
from h2o.automl import H2OAutoML
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators import H2ODeepWaterEstimator
import platform
import sys
import time
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import numpy as np


##### Start h2o#######
#h2o.init(nthreads=71, max_mem_size='30G')
#h2o.init(ip="192.168.0.11",strict_version_check=False)
####################333
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

# data_full.to_csv(urd_path, index=False)

# Define list of pandas DataFrames for model to predict on
base_data_path = os.path.join(home_path, '0MyDataBases/40Python/J_inertia')
l_csv_test_data = []  # 'ExportFileWeather_2015.csv', 'ExportFileWeather_2014.csv', 'ExportFileWeather_2013.csv',
# 'ExportFileWeather_2012.csv', 'ExportFileWeather_2011.csv', 'ExportFileWeather_2010.csv']



# Set start and end dates for training
date_start = '2013-01-01 00:00:00'
date_end = '2016-01-01 00:00:00'

# Find row indices of training data
start_row = data_full[data_full['Time'] == date_start].index.tolist()[0]
end_row = data_full[data_full['Time'] == date_end].index.tolist()[0]

# Create training data slice and convert to training and validation H2OFrames

############################################################################333
## Starting h2o part
# h2o.init(nthreads=11)
h2o.init()


pd_train = data_full[start_row:end_row].copy()

pd_test = data_full[end_row:].copy()


train = h2o.H2OFrame(pd_train,
                     column_types=['time', 'real', 'real', 'real','real'],
                     destination_frame='Training_Validation_Frame')
training, validation = train.split_frame(ratios=[0.8])

test=h2o.H2OFrame(pd_test,
                     column_types=['time', 'real', 'real', 'real','real'],
                     destination_frame='Test_Frame')
# Define predictors and response
predictors = ['Time','Wind','Load', 'Gas Price']#  ,'if_special']
response = 'Inertia'



##############################################
#### Automl


# Run AutoML for 30 seconds
aml = H2OAutoML(max_runtime_secs = 1800)
aml.train(x=predictors, y=response, training_frame= train, validation_frame=validation, leaderboard_frame = test)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb
saved_model =h2o.save_model(aml.leader,'models/aml_leader')
 # 'C:\\0MyDataBases\\40Python\\J_inertia\\models\\aml_leader\\GBM_grid_0_AutoML_20170910_162254_model_3'
#u'C:\\0MyDataBases\\40Python\\J_inertia\\models\\aml_leader\\GBM_grid_0_AutoML_20170911_081932_model_0' rmse 22525

load_aml2=h2o.load_model('C:\\0MyDataBases\\40Python\\J_inertia\\models\\aml_leader\\GBM_grid_0_AutoML_20170911_081932_model_0')
perf_aml2=load_aml2.model_performance(test_data=test)
perf_aml2

### loading the stored model
load_aml=h2o.load_model('C:\\0MyDataBases\\40Python\\J_inertia\\models\\gbm_automl')
perf_aml=load_aml.model_performance(test_data=test)
perf_aml

#
# pred_aml=load_aml.predict(test)

#### testing ################################################################

predict=pd_test.copy()
#
# # predict['predict_dnn']=pred.as_data_frame()['predict']
# predict['predict_aml']=pred_aml.as_data_frame()['predict'].values
#





#################################################################
##### working with the loaded model
# load_dnn=h2o.load_model('models/dnn_inertmed/inertia_first_try_22767')
# perf_load=load_dnn.model_performance(test_data=test)
# perf_load
#
# # Get model predictions on test data, put directly in pandas DataFrames inside dictionary
# pred_dnn_load = load_dnn.predict(test_data=test)
#
# predict['predict_dnn_load']=pred_dnn_load.as_data_frame()['predict'].values

#####################################################################

no_rows=len(training)
#####################################################################
# Run DNN
model = H2ODeepLearningEstimator(model_id='inertia_second_try'
                                 # ,checkpoint= load_dnn #model #'inertia_first_try' #load_dnn
                                 ,standardize=True
                                 ,epochs=1000
                                 ,hidden=[150,150,150]
                                 ,activation= "tanh" #""rectifier" # "maxoutwithdropout" #"rectifierwithdropout" #
                                 # , hidden_dropout_ratios=[0.15, 0.15, 0.15]
                                 , max_w2=10.0
                                 ,stopping_metric='RMSE'
                                 ,regression_stop=9000
                                 , stopping_rounds = 0
                                 # ,stopping_tolerance=1e-1
                                 , l1 =0 # 1e-6
                                 , l2 =0 # 1e-6
                                 , ignore_const_cols = False
                                 #  ,stopping_tolerance=10e-20
                                 #####################################################################
                                 ###### Controlling the scoring epochs
                                 , score_interval = 0
                                 , score_duty_cycle = 1
                                 , shuffle_training_data = False  # Recommended True, but False gives better deviance
                                 , replicate_training_data = True
                                 , train_samples_per_iteration = int(100* (no_rows))
                                 ####################################################################
                                 ### more control
                                 , input_dropout_ratio=1e-5
                                 ################################# Controlling the Momentum
                                 , adaptive_rate=False
                                 # ,rho=0.999
                                 # ,epsilon=1e-10
                                 , rate=0.005  # 0.000004 # Default is 0.005; 0.00005 is too smooth enough ?
                                 , rate_annealing=1e-60 # Default is 1e-6
                                 # ################
                                 # # ,rate_decay=
                                 # # , momentum_start=0
                                 # # , momentum_ramp=1e10
                                 # # , momentum_stable=1e10
                                 # # , nesterov_accelerated_gradient=False
                                 # , initial_weight_distribution="UniformAdaptive"
                                 # "normal", "Uniform", "UniformAdaptive"
                                 ,overwrite_with_best_model=True
                                )

model.train(x=predictors, y=response, training_frame=training, validation_frame=validation)


perf=model.model_performance(test_data=test)
perf
# Get model predictions on test data, put directly in pandas DataFrames inside dictionary
pred_dnn = model.predict(test_data=test)



# predict=pd_test.copy()
predict['predict_dnn']=pred_dnn.as_data_frame()['predict'].values


##################################################3
# Save DNN model to /tmp folder
h2o.save_model(model, path='models/dnn_inertmed', force=True)

#################################################################
##### working with the loaded model
load_dnn=h2o.load_model('models/dnn_inertmed/inertia_first_try_22767')
perf_load=load_dnn.model_performance(test_data=test)
perf_load

# Get model predictions on test data, put directly in pandas DataFrames inside dictionary
pred_dnn_load = load_dnn.predict(test_data=test)


predict['predict_dnn_load']=pred_dnn_load.as_data_frame()['predict'].values

# plotting plotly




# N = 500
# random_x = np.linspace(0, 1, N)
# random_y = np.random.randn(N)
inertia = go.Scatter(
    x = predict['Time'],
    y = predict['Inertia'],
    name='original inertia'
)

# prediction_aml = go.Scatter(
#     x = predict['Time'],
#     y = predict['predict_aml'],
#     name='prediction_aml'
# )

prediction_dnn = go.Scatter(
    x = predict['Time'],
    y = predict['predict_dnn'],
    name='prediction_dnn'
)

prediction_dnn_load = go.Scatter(
    x = predict['Time'],
    y = predict['predict_dnn_load'],
    name='prediction_dnn_load'
)

error_aml = go.Scatter(
    x = predict['Time'],
    y = predict['Inertia']- predict['predict_aml'],
    name='error_aml'
)
error_dnn = go.Scatter(
    x = predict['Time'],
    y = predict['Inertia']- predict['predict_dnn'],
    name='error_dnn'
)
error_dnn_load = go.Scatter(
    x = predict['Time'],
    y = predict['Inertia']- predict['predict_dnn_load'],
    name='error_dnn_load'
)


data = [inertia,prediction_aml,prediction_dnn,prediction_dnn_load, error_aml,error_dnn,error_dnn_load]

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

