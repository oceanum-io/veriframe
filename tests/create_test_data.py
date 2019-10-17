"""" Create test data for tests """

import numpy as np
import datetime
import pandas as pd

def create_test_data(n=100, nmod=1):
        obs = np.linspace(0,3,n)
        d = {'obs': obs}
        # Models
        for i in range(1,nmod+1):
            d['m%i' % i] = obs * 1.2 * i # 0.8*data + .1*np.random.randn(len(x))
        t0 = datetime.datetime.today()
        time = [t0 + datetime.timedelta(hours=x) for x in range(0, n)]
        return pd.DataFrame(data=d, index=time)

def create_test_data_sin(n=100, nmod=1):
        obs = np.sin(np.linspace(0,4*np.pi,n))
        d = {'obs': obs}
        # Models
        for i in range(1,nmod+1):
            d['m%i' % i] = obs * 1.2 * i # 0.8*data + .1*np.random.randn(len(x))
        t0 = datetime.datetime.today()
        time = [t0 + datetime.timedelta(hours=x) for x in range(0, n)]
        return pd.DataFrame(data=d, index=time)
