import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from data_preparation import get_dataframes_with_amount_per_cluster
import joblib


def regression_feature_importance(df, features):
    X = df[features]
    y = df['Amount']

    model = LinearRegression()
    model.fit(X, y)
    feature_importances = model.coef_
    feature_importances = [abs(importance) for importance in feature_importances]

    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importances)
    plt.xlabel('Importance')
    plt.show()


def linear_regression(df, features):
    X = df[features]
    y = df['Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle=True)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print(f'Average number of rides per hour in cluster {df["cluster"].unique()}: ', len(df)/24)
    print('Average prediction error per hour: ', mae)

    return mae


def get_average_error_among_all_clusters_with_linear_regression(df, features):
    cluster_0 = get_dataframes_with_amount_per_cluster(df, 0)
    cluster_1 = get_dataframes_with_amount_per_cluster(df, 1)
    cluster_2 = get_dataframes_with_amount_per_cluster(df, 2)
    cluster_3 = get_dataframes_with_amount_per_cluster(df, 3)
    cluster_4 = get_dataframes_with_amount_per_cluster(df, 4)
    cluster_5 = get_dataframes_with_amount_per_cluster(df, 5)
    cluster_6 = get_dataframes_with_amount_per_cluster(df, 6)
    cluster_7 = get_dataframes_with_amount_per_cluster(df, 7)
    cluster_8 = get_dataframes_with_amount_per_cluster(df, 8)
    cluster_9 = get_dataframes_with_amount_per_cluster(df, 9)

    cluster_dfs = [cluster_0, cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6, cluster_7, cluster_8, cluster_9]
    maes = []

    for dataframe in cluster_dfs:
        maes.append(linear_regression(dataframe, features))

    print('Average prediction error per hour per cluster: ', sum(maes)/10)
    print('Average rides per hour per cluster: ', len(df)/(24*10))


def random_forest_feature_importance(df, features, max_depth):
    X = df[features]
    y = df['Amount']

    model = RandomForestRegressor(max_depth=max_depth)
    model.fit(X, y)
    feature_importances = model.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importances)
    plt.xlabel('Importance')
    plt.savefig(f'Feature importance cluster {str(df["cluster"].unique()).strip("[]")}')
    plt.close()


def random_forest_regression_parameter_tuning(df, features):
    X = df[features]
    y = df['Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle=True)

    max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    max_depth_mean_errors = []

    for value in max_depth_values:
        model = RandomForestRegressor(max_depth=value)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        max_depth_mean_errors.append(mae)

    plt.plot(max_depth_values, max_depth_mean_errors)
    plt.xlabel('Max_depth')
    plt.ylabel('Mean absolute error')
    plt.savefig(f'Parameter tuning cluster {str(df["cluster"].unique()).strip("[]")}')
    plt.close()


def random_forest_regression(df, features, max_depth, export):
    X = df[features]
    y = df['Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle=True)
    model = RandomForestRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)
    if export:
        joblib.dump(model, f'cluster_{str(df["cluster"].unique()).strip("[]")}_regressor.joblib')
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Average number of rides per hour in cluster {str(df["cluster"].unique()).strip("[]")}: ', len(df) / 24)
    print('Average prediction error per hour:', mae)

    return mae


def get_average_error_among_all_clusters_with_random_forest_regression(df):
    cluster_0 = get_dataframes_with_amount_per_cluster(df, 0)
    cluster_1 = get_dataframes_with_amount_per_cluster(df, 1)
    cluster_2 = get_dataframes_with_amount_per_cluster(df, 2)
    cluster_3 = get_dataframes_with_amount_per_cluster(df, 3)
    cluster_4 = get_dataframes_with_amount_per_cluster(df, 4)
    cluster_5 = get_dataframes_with_amount_per_cluster(df, 5)
    cluster_6 = get_dataframes_with_amount_per_cluster(df, 6)
    cluster_7 = get_dataframes_with_amount_per_cluster(df, 7)
    cluster_8 = get_dataframes_with_amount_per_cluster(df, 8)
    cluster_9 = get_dataframes_with_amount_per_cluster(df, 9)

    mae_cluster_0 = random_forest_regression(cluster_0, ['Hour'], 3, False)
    mae_cluster_1 = random_forest_regression(cluster_1, ['Hour'], 3, False)
    mae_cluster_2 = random_forest_regression(cluster_2, ['Hour'], 1, False)
    mae_cluster_3 = random_forest_regression(cluster_3, ['Hour'], 1, False)
    mae_cluster_4 = random_forest_regression(cluster_4, ['Hour'], 1, False)
    mae_cluster_5 = random_forest_regression(cluster_5, ['Hour'], 1, False)
    mae_cluster_6 = random_forest_regression(cluster_6, ['Hour'], 1, False)
    mae_cluster_7 = random_forest_regression(cluster_7, ['Hour'], 3, False)
    mae_cluster_8 = random_forest_regression(cluster_8, ['Hour'], 3, False)
    mae_cluster_9 = random_forest_regression(cluster_9, ['Hour'], 1, False)

    maes = [mae_cluster_0, mae_cluster_1, mae_cluster_2, mae_cluster_3, mae_cluster_4, mae_cluster_5, mae_cluster_6, mae_cluster_7, mae_cluster_8, mae_cluster_9]
    print('Average prediction error per hour per cluster: ', sum(maes) / 10)
    print('Average rides per hour per cluster: ', len(df)/(24*10))
