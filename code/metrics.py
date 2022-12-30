import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def R2(pred, true):
    # Calculate the mean of the true values
    mean_true = true.mean()

    # Calculate the sum of squares of the residuals
    ss_residual = ((true - pred) ** 2).sum()

    # Calculate the total sum of squares
    ss_total = ((true - mean_true) ** 2).sum()

    # Calculate the R^2 score
    r2 = 1 - (ss_residual / ss_total)

    return r2

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    r2 = R2(pred, true)
    rse = RSE(pred, true)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return r2, rse, mae, mse, rmse, mape, mspe
