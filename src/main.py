import numpy as np
from regression import Regression
import logging
logging.basicConfig(level=logging.DEBUG, filename='../log/regression.log', filemode='w')

def run():
    r = Regression()

    guess = np.array([5, 5, 2])
    logging.info(r.bayes_estimation(guess))

    exog = r.data_frame[['constant', 'x']]
    logging.info(r.statistical_estimation(exog))


if __name__ == '__main__':
    run()
