import joblib
import numpy as np
import matplotlib.pyplot as plt


def daily_demand_prediction_per_hour(model_name):
    model = joblib.load(model_name)

    hours = range(0, 24)
    hours = np.array(hours).reshape(-1, 1)

    predictions = model.predict(hours)
    predictions = [int(p) for p in predictions]

    plt.plot(hours, predictions)
    plt.xlabel('Hours')
    plt.ylabel('Demand')
    plt.savefig(f'Demand prediction for cluster {model_name[8]}')
    plt.close()
