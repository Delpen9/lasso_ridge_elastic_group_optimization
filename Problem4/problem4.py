# Standard Libraries
import os
import numpy as np

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Load the .mat file
import scipy.io

def single_observation_graph(
    X : np.ndarray,
    filename : str,
    sensor_number : int
) -> None:
    '''
    '''
    current_path = os.path.abspath(__file__)
    file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', filename))

    color_palette = sns.color_palette('hls', 10)

    for i in range(X.shape[0]):
        sns.lineplot(
            x = np.arange(0, X.shape[1]),
            y = X[i],
            color = color_palette[i % 10]
        )

    plt.title(fr'Sensor {str(sensor_number)}: Observations per Timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Observation Value')

    plt.savefig(file_path, dpi = 100)
    plt.clf()
    plt.cla()

def make_observation_graphs(
    X : np.ndarray
) -> None:
    '''
    '''
    sensors = np.arange(0, 10)
    for sensor in sensors:
        single_observation_graph(
            X[sensor],
            filename = fr'sensor_{sensor}_observation',
            sensor_number = sensor
        )

def b_spline_reduction(
    X : np.ndarray
) -> None:
    '''
    '''
    return None

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    data_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'NSC.mat'))
    data_train = scipy.io.loadmat(data_file_path)

    X_train = np.stack(data_train['x'][0])
    make_observation_graphs(X_train)

    data_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'NSC.test.mat'))
    data_test = scipy.io.loadmat(data_file_path)
    X_test = np.stack(data_test['x_test'][0])
    # print(X_test)