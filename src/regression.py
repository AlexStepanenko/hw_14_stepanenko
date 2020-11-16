import numpy as np, pandas as pd
import scipy
import statsmodels.api as sm
from scipy.optimize import minimize


class Regression:
    def __init__(self):
        N = 20
        self.x_axis = np.linspace(0, 20, N)
        noise = np.random.normal(loc=0.0, scale=2.0, size=N)
        self.y_axis = 3 * self.x_axis + 1 + noise
        self.data_frame = pd.DataFrame({'y': self.y_axis, 'x': self.x_axis})
        self.data_frame['constant'] = 0

    def bayes_estimation(self, guess):
        return minimize(
            self.MLERegression,
            guess,
            method='Nelder-Mead',
            options={'disp': True}
        )

    def statistical_estimation(self, exog):
        return sm.OLS(self.y_axis, exog).fit().summary()

    def MLERegression(self, params):
        intercept, beta, sd = params[0], params[1], params[2]
        yhat = intercept + beta*self.x_axis
        negLL = -np.sum(scipy.stats.norm.logpdf(self.y_axis, loc=yhat, scale=sd))
        return negLL
