import pandas as pd
from datetime import datetime
from tqdm import tqdm
from highcharts import Highchart
import os


def aggregate_by_day_month_year(dataframe, aggregations, date_column_name='Date'):
    """Aggregates pandas DataFrames by day, month, and year using indices.

    Args:
        dataframe (pandas DataFrame): DataFrame with column that can be converted to pandas DatetimeIndex.
        aggregations (set): set of strings defining which aggregations (Yearly, Monthly, Daily) to use.
        date_column_name (string): Name of dataframe column to be converted to DatetimeIndex.

    Returns:
        dictionary: Maps words 'Daily', 'Monthly', and 'Yearly' to aggregated pandas DataFrame.

    """

    # Initialize dictionary to be returned
    return_dict = {}

    # Create time index
    times = pd.DatetimeIndex(dataframe[date_column_name])

    # Create daily aggregate with error column
    if 'Daily' in aggregations:
        pd_daily = dataframe.groupby([times.year, times.month, times.day]).sum() # Aggregate by day
        pd_daily.reset_index(inplace=True)  # Turns multi index into columns
        pd_daily = pd_daily.rename(columns={'level_0': 'Year', 'level_1': 'Month', 'level_2': 'Day'})
        pd_daily['Date'] = pd_daily.apply(lambda row:
                                          datetime(int(row['Year']), int(row['Month']), int(row['Day']), 1), axis=1)
        pd_daily['Error'] = pd_daily['Prediction'] - pd_daily['ACT']

        return_dict['Daily'] = pd_daily

    # Create monthly aggregate with error column
    if 'Monthly' in aggregations:
        pd_monthly = dataframe.groupby([times.year, times.month]).sum() # Aggregate by month
        pd_monthly.reset_index(inplace=True)  # Turns multi index into columns
        pd_monthly = pd_monthly.rename(columns={'level_0': 'Year', 'level_1': 'Month'})  # Rename index columns
        pd_monthly['Date'] = pd_monthly.apply(lambda row: datetime(int(row['Year']), int(row['Month']), 1), axis=1)
        pd_monthly['Error'] = pd_monthly['Prediction'] - pd_monthly['ACT']

        return_dict['Monthly'] = pd_monthly

    # Create yearly aggregate with error column
    if 'Yearly' in aggregations:
        pd_yearly = dataframe.groupby([times.year]).sum()
        pd_yearly.reset_index(inplace=True)
        pd_yearly = pd_yearly.rename(columns={'index': 'Date'})
        pd_yearly['Error'] = pd_yearly['Prediction'] - pd_yearly['ACT']

        return_dict['Yearly'] = pd_yearly

    return return_dict


def make_inverted_error_highcharts(predictions, folder_path):
    """Creates htmls with actual, prediction, and error plots with yearly aggregation.

    Args:
        actual_data (dictionary): Real data with structure {Aggregation: pandas DataFrame}.
        predictions (dictionary): Predictions with structure {Year: {Aggregation: pandas DataFrame}}.
        folder_path (string): Path to folder where html files are to be saved.

    """

    # Define validation years to check error with

    # Create dictionary mapping year predicted upon to list of lists of applied weather years and prediction errors
    inverted_dict = {}
    for applied_weather_year in sorted(predictions):
        df = predictions[applied_weather_year]['Yearly']
        for predicted_year in df.Date:
            if int(applied_weather_year) > 2016:
                try:
                    inverted_dict[str(predicted_year)].append([str(int(applied_weather_year) + predicted_year - 2015), float(df.loc[df['Date'] == int(predicted_year)].Prediction)])
                except KeyError:
                    inverted_dict[str(predicted_year)] = [[str(int(applied_weather_year) + predicted_year - 2015), float(df.loc[df['Date'] == int(predicted_year)].Prediction)]]

    # Create inverted error plots
    for year in sorted(inverted_dict):

        # Define chart dimensions
        H = Highchart(width=1920, height=800)

        # Initialize options
        options = {
            'chart': {
                'type': 'column'
            },
            'title': {
                'text': 'URD Fault Prediction Error for {} with offset weather applied (Only for years in validation set)'.format(year)
            },
            'subtitle': {
                'text': 'Click the links above to change the year predicted on.'
            },
            'xAxis': {
                'type': 'category',
                'title': {
                    'text': ''
                }
            },
            'yAxis': [{
                'gridLineWidth': 0,
                'title': {
                    'text': '',
                    'style': {
                        'color': 'Highcharts.getOptions().colors[1]'
                    }
                },
                'labels': {
                    'format': '{value}',
                    'style': {
                        'color': 'Highcharts.getOptions().colors[1]'
                    }
                },
                'opposite': False
            }],
            'tooltip': {
                'shared': True,
                'pointFormat': '{series.name}: <b>{point.y:.0f}</b> <br />'

            },
            'legend': {
                'layout': 'vertical',
                'align': 'left',
                'x': 80,
                'verticalAlign': 'top',
                'y': 55,
                'floating': True,
                'backgroundColor': "(Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF'"
            },
            'plotOptions': {
                'series': {
                    'borderWidth': 0,
                    'dataLabels': {
                        'enabled': False,
                        'format': '{point.y:,.0f}',
                        'formatter': 'function() {if (this.y != 0) {return Math.round(this.y)} else {return null;}}'
                    }
                }
            },
        }

        # Plot with highcharts
        H.set_dict_options(options)
        H.add_data_set(inverted_dict[year], 'column', 'Error', dataLabels={'enabled': True}, color='#777', animation=True)

        # Export plot
        filename = os.path.join(folder_path, 'URD_Prediction_Errors_{}'.format(year))
        try:
            H.save_file(filename)
        except IOError:  # Raised if folder_path directory doesn't exist
            os.mkdir(folder_path)
            H.save_file(filename)

    # Open plot and replace beginning of file with links
    headstring = "<center> "
    for year in sorted(inverted_dict):
        filename = os.path.join(folder_path, 'URD_Prediction_Errors_{}.html'.format(year))
        headstring += '<a href="{0}.html" style="color: #555; font-size: 24px">{1}</a> &ensp;'.format(filename[:-5],
                                                                                                      year)
    headstring += " </center>"

    for year in sorted(inverted_dict):
        filename = os.path.join(folder_path, 'URD_Prediction_Errors_{}.html'.format(year))
        with open(filename, 'r') as f:
            content = f.read()
        with open(filename, 'w') as f:
            f.write(headstring)
            f.write(content)


def make_highcharts(actual_data, predictions, folder_path):
    """Creates htmls with actual, prediction, and error plots with yearly aggregation.

    Args:
        actual_data (dictionary): Real data with structure {Aggregation: pandas DataFrame}.
        predictions (dictionary): Predictions with structure {Year: {Aggregation: pandas DataFrame}}.
        folder_path (string): Path to folder where html files are to be saved.

    """

    # Convert real data to list of lists for plotting with highcharts
    actual = actual_data['Time', 'Inertial'].values.tolist()

    # Initialize list of years
    l_years = []

    # Create yearly plots
    for year in sorted(predictions):
        df = predictions[year]['Yearly']
        if year == '.Actual':
            year = 'Actual'
        l_years.append(year)

        # Define chart dimensions
        H = Highchart(width=1920, height=800)

        # Initialize options
        options = {
            'chart': {
                'type': 'column'
            },
            'title': {
                'text': 'URD Fault Predictions with {} Weather Applied to 2016 Data'.format(year)
            },
            'subtitle': {
                'text': 'Click the links above to change the weather offset.'
            },
            'xAxis': {
                'type': 'category',
                'title': {
                    'text': ''
                }
            },
            'yAxis': [{
                'gridLineWidth': 0,
                'title': {
                    'text': '',
                    'style': {
                        'color': 'Highcharts.getOptions().colors[1]'
                    }
                },
                'labels': {
                    'format': '{value}',
                    'style': {
                        'color': 'Highcharts.getOptions().colors[1]'
                    }
                },
                'opposite': False
            }],
            'tooltip': {
                'shared': True,
                'pointFormat': '{series.name}: <b>{point.y:.0f}</b> <br />'

            },
            'legend': {
                'layout': 'vertical',
                'align': 'left',
                'x': 80,
                'verticalAlign': 'top',
                'y': 55,
                'floating': True,
                'backgroundColor': "(Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF'"
            },
            'plotOptions': {
                'series': {
                    'borderWidth': 0,
                    'dataLabels': {
                        'enabled': False,
                        # 'format': '{point.y:,.0f}',
                        'formatter': 'function() {if (this.y != 0) {return Math.round(this.y)} else {return null;}}'
                    }
                }
            },
        }

        # Convert pandas dataframe to lists of lists for plotting with highcharts
        error = df.reset_index()[['Date', 'Error']].values.tolist()
        prediction = df.reset_index()[['Date', 'Prediction']].values.tolist()

        # Plot with highcharts
        H.set_dict_options(options)
        H.add_data_set(actual, 'line', 'Actual', marker={'enabled': False}, color='#A00', animation=False)
        H.add_data_set(prediction, 'line', 'Prediction', marker={'enabled': False}, dashStyle='dash', color='#00A', animation=False)
        H.add_data_set(error, 'column', 'Error', dataLabels={'enabled': True}, color='#777', animation=False)

        # Export plot
        filename = os.path.join(folder_path, 'URD_Prediction_{}_Weather'.format(year))
        try:
            H.save_file(filename)
        except IOError:  # Raised if folder_path directory doesn't exist
            os.mkdir(folder_path)
            H.save_file(filename)

    # Open plot and replace beginning of file with links
    headstring = "<center> "
    for year in l_years:
        filename = os.path.join(folder_path, 'URD_Prediction_{}_Weather.html'.format(year))
        headstring += '<a href="{0}.html" style="color: #555; font-size: 24px">{1}</a> &ensp;'.format(filename[:-5], year)
    headstring += " </center>"

    for year in l_years:
        filename = os.path.join(folder_path, 'URD_Prediction_{}_Weather.html'.format(year))
        with open(filename, 'r') as f:
            content = f.read()
        with open(filename, 'w') as f:
            f.write(headstring)
            f.write(content)


def visualize_urd_highcharts(real_data, predictions, folder_path, aggregations={'Yearly'}):
    """Creates html files to visualize real data and predictions for URD data using highcharts

        Args:

        Returns:

    """

    # Create dictionary of aggregation type to actual dataframe
    d_actual = aggregate_by_day_month_year(real_data, aggregations, 'Time')

    # Create nested dictionary of applied weather year to aggregation type to prediction dataframe
    d_predictions = {}
    print("Aggregating data...")
    for prediction_year in tqdm(predictions):
        d_predictions[prediction_year] = aggregate_by_day_month_year(predictions[prediction_year],
                                                                     aggregations, 'Time')

    make_highcharts(d_actual, d_predictions, folder_path)
    make_inverted_error_highcharts(d_predictions, folder_path)
