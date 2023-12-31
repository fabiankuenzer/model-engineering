from data_ingestion import load_data
from data_exploration import *
from data_preparation import *
from clustering import *
from regression import *
from predict import *
import warnings
warnings.simplefilter(action='ignore', category=Warning)

raw_data = load_data()
sample_size = 10000  # only for quick presentation purposes, set to 4534327 for complete data
sample_relation = len(raw_data)/sample_size
raw_data = raw_data.sample(sample_size)

# Data understanding
# print(raw_data.info())
# print(raw_data.describe())
# show_date_time_value_range(raw_data)
#
# Explorative data analysis
# plot_pickups_per_base(raw_data)
# plot_pickup_frequency_per_longitude(raw_data)
# plot_pickup_frequency_per_latitude(raw_data)
# plot_share_of_pickups_per_month(raw_data)
# plot_share_of_pickups_per_day(raw_data)
# plot_share_of_pickups_per_hour(raw_data)
# plot_share_of_pickups_per_minute(raw_data)
# plot_pickups_on_map(raw_data, sample_size)
# plot_pickups_on_map_per_base(raw_data, 'B02617')
# plot_pickups_on_map_per_base(raw_data, 'B02598')
# plot_pickups_on_map_per_base(raw_data, 'B02682')
# plot_pickups_on_map_per_base(raw_data, 'B02764')
# plot_pickups_on_map_per_base(raw_data, 'B02512')

# Data preparation
print('PREPARE DATA: START')
filtered_df = remove_pickups_outside_of_nyc(raw_data)
# plot_pickups_on_map(filtered_df)
final_df = append_weekday_column(filtered_df)
final_df = append_date_time_columns(final_df)
final_df = append_weekend_column(final_df)
final_df = label_encode_column(final_df, 'Base')
print('PREPARE DATA: DONE')

# Modelling
# Clustering
print('CLUSTER ALGORITHM AND FEATURE PERFORMANCE COMPARISON')
clustering_features = ['Lat', 'Lon', 'Weekday', 'Base', 'Month', 'Hour', 'Day', 'Minute', 'Weekend']
feature_sub_lists = get_sub_lists(clustering_features)[:-1]
for sub_list in feature_sub_lists:
    kmeans_clustering(final_df, sub_list, False)
for sub_list in feature_sub_lists:
    agglomerative_clustering(final_df, sub_list)
print('CHOOSE KMEANS')

clustered_df, centroids_longitude, centroids_latitude = kmeans_clustering(final_df, ['Lat', 'Lon'], True)
cluster_center_coordinates = list(zip(centroids_latitude, centroids_longitude))
print('Cluster center coordinates:')
for c in cluster_center_coordinates:
    print(c)
plot_clusters_on_map(clustered_df, centroids_longitude, centroids_latitude)

# Regression
cluster_0 = get_dataframes_with_amount_per_cluster(clustered_df, 0, sample_relation)
cluster_1 = get_dataframes_with_amount_per_cluster(clustered_df, 1, sample_relation)
cluster_2 = get_dataframes_with_amount_per_cluster(clustered_df, 2, sample_relation)
cluster_3 = get_dataframes_with_amount_per_cluster(clustered_df, 3, sample_relation)
cluster_4 = get_dataframes_with_amount_per_cluster(clustered_df, 4, sample_relation)
cluster_5 = get_dataframes_with_amount_per_cluster(clustered_df, 5, sample_relation)
cluster_6 = get_dataframes_with_amount_per_cluster(clustered_df, 6, sample_relation)
cluster_7 = get_dataframes_with_amount_per_cluster(clustered_df, 7, sample_relation)
cluster_8 = get_dataframes_with_amount_per_cluster(clustered_df, 8, sample_relation)
cluster_9 = get_dataframes_with_amount_per_cluster(clustered_df, 9, sample_relation)

print('LINEAR REGRESSION FEATURE IMPORTANCE: START')
regression_feature_importance(cluster_0, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'])
regression_feature_importance(cluster_1, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'])
regression_feature_importance(cluster_2, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'])
regression_feature_importance(cluster_3, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'])
regression_feature_importance(cluster_4, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'])
regression_feature_importance(cluster_5, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'])
regression_feature_importance(cluster_6, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'])
regression_feature_importance(cluster_7, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'])
regression_feature_importance(cluster_8, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'])
regression_feature_importance(cluster_9, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'])
print('LINEAR REGRESSION FEATURE IMPORTANCE: DONE')

print('LINEAR REGRESSION PERFORMANCE: START')
linear_regression(cluster_0, ['Lat', 'Lon', 'Weekend'], False)
linear_regression(cluster_1, ['Lat', 'Lon', 'Hour'], False)
linear_regression(cluster_2, ['Lat', 'Lon', 'Hour'], False)
linear_regression(cluster_3, ['Lat', 'Lon', 'Hour'], False)
linear_regression(cluster_4, ['Lat', 'Lon', 'Hour'], False)
linear_regression(cluster_5, ['Lat', 'Lon', 'Hour'], False)
linear_regression(cluster_6, ['Lat', 'Lon', 'Hour'], False)
linear_regression(cluster_7, ['Lat', 'Lon', 'Hour'], False)
linear_regression(cluster_8, ['Lat', 'Lon', 'Hour'], False)
linear_regression(cluster_9, ['Lat', 'Lon', 'Hour'], False)
print('LINEAR REGRESSION PERFORMANCE: DONE')

print('FINAL LINEAR REGRESSION FEATURE IMPORTANCE: START')
get_average_error_among_all_clusters_with_linear_regression(clustered_df, ['Lat', 'Lon', 'Hour'], sample_relation)
print('FINAL LINEAR REGRESSION FEATURE IMPORTANCE: DONE')

print('RANDOM FOREST REGRESSOR FEATURE IMPORTANCE: START')
random_forest_feature_importance(cluster_0, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'], 4)
random_forest_feature_importance(cluster_1, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'], 4)
random_forest_feature_importance(cluster_2, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'], 4)
random_forest_feature_importance(cluster_3, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'], 4)
random_forest_feature_importance(cluster_4, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'], 4)
random_forest_feature_importance(cluster_5, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'], 4)
random_forest_feature_importance(cluster_6, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'], 4)
random_forest_feature_importance(cluster_7, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'], 4)
random_forest_feature_importance(cluster_8, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'], 4)
random_forest_feature_importance(cluster_9, ['Lat', 'Lon', 'Base', 'Weekday', 'Month', 'Day', 'Hour', 'Minute', 'Weekend'], 4)
print('RANDOM FOREST REGRESSOR FEATURE IMPORTANCE: DONE')

print('RANDOM FOREST REGRESSOR PARAMETER TUNING: START')
random_forest_regression_parameter_tuning(cluster_0, ['Hour'])
random_forest_regression_parameter_tuning(cluster_1, ['Hour'])
random_forest_regression_parameter_tuning(cluster_2, ['Hour'])
random_forest_regression_parameter_tuning(cluster_3, ['Hour'])
random_forest_regression_parameter_tuning(cluster_4, ['Hour'])
random_forest_regression_parameter_tuning(cluster_5, ['Hour'])
random_forest_regression_parameter_tuning(cluster_6, ['Hour'])
random_forest_regression_parameter_tuning(cluster_7, ['Hour'])
random_forest_regression_parameter_tuning(cluster_8, ['Hour'])
random_forest_regression_parameter_tuning(cluster_9, ['Hour'])
print('RANDOM FOREST REGRESSOR PARAMETER TUNING: DONE')

print('RANDOM FOREST REGRESSOR PERFORMANCE: START')
random_forest_regression(cluster_0, ['Hour'], 4, True)
random_forest_regression(cluster_1, ['Hour'], 4, True)
random_forest_regression(cluster_2, ['Hour'], 4, True)
random_forest_regression(cluster_3, ['Hour'], 4, True)
random_forest_regression(cluster_4, ['Hour'], 4, True)
random_forest_regression(cluster_5, ['Hour'], 4, True)
random_forest_regression(cluster_6, ['Hour'], 4, True)
random_forest_regression(cluster_7, ['Hour'], 4, True)
random_forest_regression(cluster_8, ['Hour'], 4, True)
random_forest_regression(cluster_9, ['Hour'], 4, True)
print('RANDOM FOREST REGRESSOR PERFORMANCE: DONE')

print('FINAL RANDOM FOREST REGRESSOR PERFORMANCE: START')
get_average_error_among_all_clusters_with_random_forest_regression(clustered_df, sample_relation)
print('FINAL RANDOM FOREST REGRESSOR PERFORMANCE: DONE')


print('PRODUCE PREDICTIONS WITH RANDOM FOREST REGRESSOR: START')
daily_demand_prediction_per_hour('cluster_0_regressor.joblib')
daily_demand_prediction_per_hour('cluster_1_regressor.joblib')
daily_demand_prediction_per_hour('cluster_2_regressor.joblib')
daily_demand_prediction_per_hour('cluster_3_regressor.joblib')
daily_demand_prediction_per_hour('cluster_4_regressor.joblib')
daily_demand_prediction_per_hour('cluster_5_regressor.joblib')
daily_demand_prediction_per_hour('cluster_6_regressor.joblib')
daily_demand_prediction_per_hour('cluster_7_regressor.joblib')
daily_demand_prediction_per_hour('cluster_8_regressor.joblib')
daily_demand_prediction_per_hour('cluster_9_regressor.joblib')
print('PRODUCE PREDICTIONS WITH RANDOM FOREST REGRESSOR: DONE')
print('VIEW PRODUCED VISUALIZATIONS AND MODELS: ls')
