import numpy as np
from hurst import compute_Hc
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def time_series_half_life(ts):
    """ 
    Calculates the half life of a mean reversion
    
    np.linalg.lstsq => Linear Solver for B = Ax
    """    
    ts = np.asarray(ts)        

    delta_ts = np.diff(ts)        
    lag_ts = np.vstack([ts[1:], np.ones(len(ts[1:]))]).T    
    beta = np.linalg.lstsq(lag_ts, delta_ts)    
    return (np.log(2) / beta[0])[0]

def time_series_hurst_exponent(ts):
    """ 
    Calculates hurst exponent of a time series    
        H = 0.5 => Random
        H > 0.5 => Trendnig
        H < 0.5 => Mean Reverting
    """    
    return compute_Hc(np.abs(ts), kind='price', simplified=True)[0]

def time_series_hurst_exponent_v2(series, min_lag=10, max_lag=None):
    """
    Calculate Hurst Exponent using Rescaled Range (R/S) Analysis
    H < 0.5 = mean reversion, H = 0.5 = random walk, H > 0.5 = trending
    """
    if max_lag is None:
        max_lag = len(series) // 2
    
    series = np.asarray(series, dtype=np.float64)
    lags = range(min_lag, max_lag)
    rs_values = []
    
    for n in lags:
        # Split series into chunks of size n
        num_chunks = len(series) // n
        rs_list = []
        
        for i in range(num_chunks):
            chunk = series[i*n:(i+1)*n]
            
            # Mean-adjusted series
            mean_chunk = np.mean(chunk)
            Y = np.cumsum(chunk - mean_chunk)
            
            # Range
            R = np.max(Y) - np.min(Y)
            
            # Standard deviation
            S = np.std(chunk, ddof=1)
            
            # Avoid division by zero
            if S > 0:
                rs_list.append(R / S)
        
        # Average R/S for this lag
        if rs_list:
            rs_values.append(np.mean(rs_list))
    
    # Log-log regression: log(RS) = H * log(n) + c
    lags_array = np.array(list(range(min_lag, min_lag + len(rs_values))))
    rs_values = np.array(rs_values)
    
    poly = np.polyfit(np.log(lags_array), np.log(rs_values), 1)
    hurst = poly[0]
    
    # Constrain to valid range [0, 1]
    return np.clip(hurst, 0, 1)

def time_series_coint_johansen(ts, normalize=True, ci=0.99):
    """ 
        Conduct johansen test on multiple time series

        function returns the maximum rank of both trace and max eigenvalues test statistics, i.e. if either one of the test concludes higher rank, we use that as the rank        

        Args:
            normalize: true will normalize the output eigenvector by sum of all positive numbers
            ci: confidence interval of the test, options are = [0.9,0.95,0.99]
    """    

    ci_arr = [0.9, 0.95, 0.99]
    ci_idx = ci_arr.index(ci)

    res = coint_johansen(ts, det_order=0, k_ar_diff=1)

    lr1_pass = res.lr1 > res.cvt[:,ci_idx]
    lr2_pass = res.lr2 > res.cvm[:,ci_idx]
    lr1_r = 1+np.argmax(lr1_pass) if np.sum(lr1_pass) > 0 else 0
    lr2_r = 1+np.argmax(lr2_pass) if np.sum(lr2_pass) > 0 else 0
    r = max(lr1_r, lr2_r)
    
    evec = res.evec[:,0]
    if normalize:
        evec = evec / evec[evec > 0].sum()    

    return evec, r

def bootstrapped(x:list|np.ndarray, n:int=-1, k:int=1000) -> list:
    '''
        Bootstrapped samples of a given array
        n: number of samples
        k: number of bootstrapped samples
    '''

    if n == -1:
        n = len(x)

    x_b = [np.mean(np.random.choice(x, size=n, replace=True)) for _ in range(k)]
    if type(x) is np.ndarray:
        x_b = np.array(x_b)

    return x_b

if __name__ == '__main__':
    print(time_series_half_life([1,2,3,3,2,1,2,3,2,1]))
