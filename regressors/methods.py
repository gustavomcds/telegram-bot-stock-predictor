# data manipulation
import pandas as pd
import numpy as np
from itertools import product

# time
from datetime import datetime, timedelta
import time

# plot
import matplotlib.pyplot as plt

# metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

#warnings
import warnings
warnings.filterwarnings('ignore')

class HoltWinters:

    """
    Holt-Winters model with the anomalies detection using Brutlag method
    
    Parameters:
    
        series(Series): initial time series
        slen(int): length of a season
        alpha(float): Holt-Winters model coefficient
        beta(float): Holt-Winters model coefficient
        gamma(float): Holt-Winters model coefficient
        n_preds(int): predictions horizon
        scaling_factor(float): sets the width of the confidence interval by Brutlag
    
    """

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96, past_dates=None, future_dates=None):

        self.series = np.array(series)
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        self.best_score = np.inf
        self.best_params = None
        self.past_dates = past_dates
        self.future_dates = future_dates

    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i + self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(
                sum(self.series[self.slen * j : self.slen * j + self.slen]) / float(self.slen)
            )
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += (
                    self.series[self.slen * j + i] - season_averages[j]
                )
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def triple_exponential_smoothing(self):
        
        """
        Make predictions using TES
        """
        
        self.result = []
        self.predictions = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:  # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])

                self.PredictedDeviation.append(0)

                self.UpperBond.append(
                    self.result[0] + self.scaling_factor * self.PredictedDeviation[0]
                )

                self.LowerBond.append(
                    self.result[0] - self.scaling_factor * self.PredictedDeviation[0]
                )
                continue

            if i >= len(self.series):  # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])
                self.predictions.append((smooth + m * trend) + seasonals[i % self.slen])

                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = (
                    smooth,
                    self.alpha * (val - seasonals[i % self.slen])
                    + (1 - self.alpha) * (smooth + trend),
                )
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = (
                    self.gamma * (val - smooth)
                    + (1 - self.gamma) * seasonals[i % self.slen]
                )
                self.result.append(smooth + trend + seasonals[i % self.slen])

                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(
                    self.gamma * np.abs(self.series[i] - self.result[i])
                    + (1 - self.gamma) * self.PredictedDeviation[-1]
                )

            self.UpperBond.append(
                self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1]
            )

            self.LowerBond.append(
                self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1]
            )

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])

        return self.predictions
    
    def calculate_mean_absolute_percentage_error(self):
        return mean_absolute_percentage_error(self.series, self.result[: len(self.series)])
    
    def calculate_root_mean_squared_error(self):
        return np.sqrt(mean_squared_error(self.series, self.result[: len(self.series)]))
    
    def plotHoltWinters(self, ticker_title, plot_intervals=False, plot_anomalies=False):
        
        """
        Plot original time series, Holt-Winters fitted time series and predictions
        
        Parameters:
        
            series(Series): dataset with timeseries
            plot_intervals(boolean): show confidence intervals
            plot_anomalies(boolean): show anomalies 
            n_points(int): how many points must be displayed in the chart
        """

        self.chart_pred = plt.figure(figsize=(20, 10))
        
        if self.past_dates is None and self.future_dates is None:
            plt.plot(self.result, label="Model")
            plt.plot(self.series, label="Actual")
        else:

            plt.plot(np.concatenate((self.past_dates, self.future_dates)), self.result, label="Model")
            plt.plot(self.past_dates, self.series, label="Actual")
           
        plt.title(f"{ticker_title}")

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(self.series))
            anomalies[self.series < self.LowerBond[: len(self.series)]] = self.series[
                self.series < self.LowerBond[: len(self.series)]
            ]
            anomalies[self.series > self.UpperBond[: len(self.series)]] = self.series[
                self.series > self.UpperBond[: len(self.series)]
            ]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

        if plot_intervals:
            plt.plot(self.UpperBond, "r--", alpha=0.5, label="Up/Low confidence")
            plt.plot(self.LowerBond, "r--", alpha=0.5)
            plt.fill_between(
                x=range(0, len(self.result)),
                y1=self.UpperBond,
                y2=self.LowerBond,
                alpha=0.2,
                color="grey",
            )

        plt.vlines(
            self.past_dates[-1],
            ymin=min(self.LowerBond),
            ymax=max(self.UpperBond),
            linestyles="dashed",
        )
        plt.axvspan(self.past_dates[-1], self.future_dates[-1], alpha=0.3, color="lightgrey")
        plt.grid(True)
        plt.axis("tight")
        plt.legend(loc="best", fontsize=13)
        
        # zooming into last 30 days + predictions period
        # plt.xlim(len(self.series) - 1 - 30, len(self.result))

        return self.chart_pred
        
def optimize_holt_winters_hyperparameters(ts, len_preds=1, slens=[365], plot=False, eval_method='rmse', allow_negative_hist=False, allow_negative_preds=False, verbose=False):
    
    """
    Predict for one Item using Triple Exponential Smoothing (TES)
    
    Parameters:
    
        ts(Series): initial time series
        len_preds(int): prediction horizon
        slens(list): list of length of season
        plot(boolean): if True, plot a chart of time series and predictions
        eval_method(str): error metric used to find best parameters (options: 'rmse' and 'mae')
        allow_negative_hist(boolean): if False, replace all negative sales values with 0
        allow_negative_preds(boolean): if False, doesn't allow any negative prediction
        verbose(boolean): if True, some status messages are shown
        
    Returns:
    
        results(list): a list with size len_preds+1 containing best error score and all predictions
    """
    
    ts = np.array(ts)
    
    if not allow_negative_hist:
        ts = np.where(ts < 0, 0, ts)
        
    alpha_vals, beta_vals, gamma_vals = (np.round(np.arange(0, 1, .1), 2) for i in range(3))
    slen_vals = np.array(slens)
    best_score = np.inf
    best_params = None
    
    if len(ts) > 8:
        best_params = (0.5, 0.5, 0.5, 4)
    else:
        best_params = (0.5, 0.5, 0.5, 1)

    for i in product(alpha_vals, beta_vals, gamma_vals, slen_vals):
        try:
            
            alpha_i, beta_i, gamma_i, slen_i = i
            model = HoltWinters(ts, slen=slen_i, alpha=alpha_i, beta=beta_i, gamma=gamma_i, n_preds=len_preds)
            model.triple_exponential_smoothing()
            
            if eval_method == 'mape':
                actual_score = model.calculate_mean_absolute_percentage_error()
            elif eval_method == 'rmse':
                actual_score = model.calculate_root_mean_squared_error()
                
            if allow_negative_preds:
                if actual_score < best_score:
                    best_score = actual_score
                    best_params = i
            else:
                if (actual_score < best_score) and (np.count_nonzero( np.array(model.result[-len_preds : ]) < 0 ) <= 0):
                    best_score = actual_score
                    best_params = i
                    
        except:
            
            pass
        
    return dict(list(zip(('alpha', 'beta', 'gamma', 'slen'), best_params)))