import sys
import json
from model import SalesPredictionLSTM
from data_utils import load_timeseries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_file = "data/sales.csv"
window_size = 6
train_test_split = 0.8
(
    x_train, y_train, x_test, y_test,
    x_test_raw, y_test_raw,
    last_window_raw, last_window
) = load_timeseries(train_file, window_size, train_test_split)

model = SalesPredictionLSTM(
    layers=[window_size, 100, 100, 1],
    dropout=0.2,
    batch_size=100,
    epochs=100,
    validation_split=0.1
)

model.build_model()

model.train(x_train, y_train)
model.save_weights('weights.h5')
