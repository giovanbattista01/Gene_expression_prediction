import numpy as np
import pyBigWig
import pandas as pd

prediction_path = '/home/vegeta/Downloads/ML4G_Project_1_Data/Landolina_GiovanBattista_Project1/gex_predicted.csv'


preds = pd.read_csv(prediction_path)['gex_predicted'].to_numpy()

#preds[np.isnan(preds)] = 0.0

nan_count = np.isnan(preds).sum()

print(f"Total number of NaN values: {nan_count}")