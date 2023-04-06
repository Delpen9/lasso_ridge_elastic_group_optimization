# Standard Libraries
import os
import numpy as np

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp

# Load the .mat file
import scipy.io

# B-Spline
from scipy.interpolate import make_interp_spline, splrep, BSpline

# Modeling
from group_lasso import GroupLasso
from sklearn.metrics import mean_squared_error

def spline_coefficients_graph(
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

    plt.title(fr'Sensor {str(sensor_number)}: Spline Coefficients')
    plt.xlabel('Index')
    plt.ylabel('Spline Coefficient')

    plt.savefig(file_path, dpi = 100)
    plt.clf()
    plt.cla()

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

def b_spline_single_row_reduction(
    row : np.ndarray
) -> np.ndarray:
    '''
    '''
    knot_count = 8

    knot_distance = row.shape[0] / (knot_count - 1)
    knots = np.arange(0, row.shape[0] + knot_distance, knot_distance)

    tck = splrep(np.arange(0, row.shape[0]), row, t = knots[1: -1])
    spline = BSpline(*tck)
    coefficients = spline.c[:-4]

    return coefficients

def b_spline_reduction(
    sensor_data : np.ndarray
) -> np.ndarray:
    '''
    '''
    reduced_sensor_data = []
    for row in sensor_data:
        reduced_sensor_data.append(b_spline_single_row_reduction(row))
    reduced_sensor_data = np.array(reduced_sensor_data)
    return reduced_sensor_data

def make_b_spline_observation_graphs(
    X : np.ndarray
) -> None:
    '''
    '''
    sensors = np.arange(0, 10)
    for sensor in sensors:
        reduced_sensor_data = b_spline_reduction(X[sensor].copy())

        spline_coefficients_graph(
            reduced_sensor_data,
            filename = fr'reduced_sensor_{sensor}_observation',
            sensor_number = sensor
        )

def reduce_data_with_bsplines(
    X : np.ndarray
) -> np.ndarray:
    '''
    '''
    sensors = np.arange(0, 10)
    X_reduced = []
    for sensor in sensors:
        reduced_sensor_data = b_spline_reduction(X[sensor].copy())
        X_reduced.append(reduced_sensor_data)
    return np.hstack(tuple(X_reduced))

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    data_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'NSC.mat'))
    data_train = scipy.io.loadmat(data_file_path)

    X_train = np.stack(data_train['x'][0])
    y_train = data_train['y']

    # make_observation_graphs(X_train)
    # make_b_spline_observation_graphs(X_train)

    X_train_reduced = reduce_data_with_bsplines(X_train)

    groups = np.repeat(np.arange(1, 11), 10)
    group_lasso = GroupLasso(
        groups = groups,
        group_reg = 0.05,
        l1_reg = 0.008,
        frobenius_lipschitz = False,
        scale_reg = 'inverse_group_size',
        subsampling_scheme = 0.5,
        supress_warning = True,
        n_iter = 5000,
        tol = 1e-10,
    )
    group_lasso.fit(X_train_reduced, y_train)

    chosen_groups = np.array(list(group_lasso.chosen_groups_)).astype(int)
    print(fr'''
    The sensors which correlate with the air/fuel ratio are:
    {chosen_groups}
    ''')

    ## ==================
    ## Plot correlations
    ## ==================
    for index in chosen_groups:
        x_train_flattened = data_train['x'][0][index].flatten().copy()
        y_train_flattened = y_train.flatten().copy()

        ax = sns.regplot(
            x = x_train_flattened,
            y = y_train_flattened,
            line_kws = {'color': 'red'},
            scatter_kws = {'alpha': 0.5, 's': 10},
            ci = None
        )

        r, p = sp.stats.pearsonr(x_train_flattened, y_train_flattened)
        ax = plt.gca()
        ax.text(
            .05,
            .8,
            'Slope (m) = {:.2f},\np-value = {:.2g}'.format(r, p),
            transform = ax.transAxes
        )

        file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', fr'sensor_{str(index)}_correlation.png'))

        plt.title(fr'Sensor {str(index)}: Observations VS. Air/Fuel Ratio')
        plt.xlabel(fr'Sensor {str(index)}: Observations')
        plt.ylabel('Air/Fuel Ratio')

        plt.savefig(file_path, dpi = 100)
        plt.clf()
        plt.cla()
    ## ==================

    data_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'NSC.test.mat'))
    data_test = scipy.io.loadmat(data_file_path)
    X_test = np.stack(data_test['x_test'][0])
    y_test = data_test['y_test']

    X_test_reduced = reduce_data_with_bsplines(X_test)

    y_pred = group_lasso.predict(X_test_reduced)

    print(fr'''
    The mean-square error on the test set is:
    {mean_squared_error(y_pred, y_test)}
    ''')