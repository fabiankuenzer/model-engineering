import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd

plt.interactive(True)


def show_date_time_value_range(df):
    month_values = []
    day_values = []
    year_values = []
    hour_values = []
    minute_values = []
    second_values = []

    for dt in df['Date/Time']:
        date_time_object = datetime.strptime(dt, '%m/%d/%Y %H:%M:%S')
        month_values.append(date_time_object.month)
        day_values.append(date_time_object.day)
        year_values.append(date_time_object.year)
        hour_values.append(date_time_object.hour)
        minute_values.append(date_time_object.minute)
        second_values.append(date_time_object.second)

    month_values = pd.Series(month_values)
    day_values = pd.Series(day_values)
    year_values = pd.Series(year_values)
    hour_values = pd.Series(hour_values)
    minute_values = pd.Series(minute_values)
    second_values = pd.Series(second_values)

    print('Years')
    print(year_values.describe())
    print('Months')
    print(month_values.describe())
    print('Days')
    print(day_values.describe())
    print('Hours')
    print(hour_values.describe())
    print('Minutes')
    print(minute_values.describe())
    print('Seconds')
    print(second_values.describe())


def plot_pickups_per_base(df):
    base1_share = df['Base'].value_counts(normalize=True).get('B02617')
    base2_share = df['Base'].value_counts(normalize=True).get('B02598')
    base3_share = df['Base'].value_counts(normalize=True).get('B02682')
    base4_share = df['Base'].value_counts(normalize=True).get('B02764')
    base5_share = df['Base'].value_counts(normalize=True).get('B02512')

    data = {'B02617': base1_share, 'B02598': base2_share, 'B02682': base3_share, 'B02764': base4_share,
            'B02512': base5_share}

    base_names = list(data.keys())
    base_percentage = list(data.values())
    base_percentage = [x * 100 for x in base_percentage]

    fig = plt.figure(figsize=(6, 5))
    plt.bar(base_names, base_percentage, width=0.4)

    plt.xlabel("Base name")
    plt.ylabel("Share of total pickups in %")
    plt.title("Share of total pickups per base")
    plt.show()


def plot_pickup_frequency_per_longitude(df):
    fig = plt.figure(figsize=(6, 5))
    plt.hist(df['Lon'], bins=200, range=(-74.3, -73.5))

    plt.xlabel("Longitude")
    plt.ylabel("Pickup frequency")
    plt.title("Pickup frequency per longitude")
    plt.show()


def plot_pickup_frequency_per_latitude(df):
    fig = plt.figure(figsize=(6, 5))
    plt.hist(df['Lat'], bins=200)

    plt.xlabel("Latitude")
    plt.ylabel("Pickup frequency")
    plt.title("Pickup frequency per latitude")
    plt.show()


def plot_share_of_pickups_per_month(df):
    month_values = []
    for dt in df['Date/Time']:
        date_time_object = datetime.strptime(dt, '%m/%d/%Y %H:%M:%S')
        month_values.append(date_time_object.month)
    month_values = pd.Series(month_values)

    fig = plt.figure(figsize=(6, 5))
    plt.bar(list(month_values.value_counts(normalize=True).sort_index().keys()),
            [x * 100 for x in month_values.value_counts(normalize=True).sort_index().to_list()], width=0.4)
    plt.xlabel("Month")
    plt.ylabel("Share in %")
    plt.title("Share of total pickups per month")
    plt.show()


def plot_share_of_pickups_per_day(df):
    day_values = []
    for dt in df['Date/Time']:
        date_time_object = datetime.strptime(dt, '%m/%d/%Y %H:%M:%S')
        day_values.append(date_time_object.day)
    day_values = pd.Series(day_values)

    fig = plt.figure(figsize=(10, 5))
    plt.bar(list(day_values.value_counts(normalize=True).sort_index().keys()),
            [x * 100 for x in day_values.value_counts(normalize=True).sort_index().to_list()], width=0.4)
    plt.xlabel("Day")
    plt.ylabel("Share in %")
    plt.title("Share of total pickups per day")
    plt.show()


def plot_share_of_pickups_per_hour(df):
    hour_values = []
    for dt in df['Date/Time']:
        date_time_object = datetime.strptime(dt, '%m/%d/%Y %H:%M:%S')
        hour_values.append(date_time_object.hour)
    hour_values = pd.Series(hour_values)

    fig = plt.figure(figsize=(10, 5))
    plt.bar(list(hour_values.value_counts(normalize=True).sort_index().keys()),
            [x * 100 for x in hour_values.value_counts(normalize=True).sort_index().to_list()], width=0.4)
    plt.xlabel("Hour")
    plt.ylabel("Share in %")
    plt.title("Share of total pickups per hour")
    plt.show()


def plot_share_of_pickups_per_minute(df):
    minute_values = []
    for dt in df['Date/Time']:
        date_time_object = datetime.strptime(dt, '%m/%d/%Y %H:%M:%S')
        minute_values.append(date_time_object.minute)
    minute_values = pd.Series(minute_values)

    fig = plt.figure(figsize=(10, 5))
    plt.bar(list(minute_values.value_counts(normalize=True).sort_index().keys()),
            [x * 100 for x in minute_values.value_counts(normalize=True).sort_index().to_list()], width=0.4)
    plt.xlabel("Minute")
    plt.ylabel("Share in %")
    plt.title("Share of total pickups per minute")
    plt.show()


def plot_pickups_on_map(df):
    geometry = [Point(xy) for xy in zip(df['Lon'], df['Lat'])]
    raw_geo_data = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    nyc_map = gpd.read_file(gpd.datasets.get_path('nybb'), crs="EPSG:4326")
    nyc_map = nyc_map.to_crs(epsg=4326)

    ax = nyc_map.plot(figsize=(7, 7), color='white', edgecolor='black')
    raw_geo_data.plot(ax=ax, markersize=1, alpha=0.1)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Pickup locations on the map of New York City")
    plt.show()


def plot_pickups_on_map_per_base(df, base):
    geometry = [Point(xy) for xy in zip(df['Lon'], df['Lat'])]
    raw_geo_data = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    geo_data_of_base = raw_geo_data[raw_geo_data['Base'] == base]
    print(geo_data_of_base.describe())

    nyc_map = gpd.read_file(gpd.datasets.get_path('nybb'), crs="EPSG:4326")
    nyc_map = nyc_map.to_crs(epsg=4326)

    ax = nyc_map.plot(figsize=(7, 7), color='white', edgecolor='black')
    geo_data_of_base.plot(ax=ax, markersize=1, alpha=0.1)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Pickup locations on the map of New York City")
    plt.show()


def plot_clusters_on_map(df, centroid_longitude, centroid_latitude):
    centroid_longitude = pd.Series(centroid_longitude, name='centroid_lon')
    centroid_latitude = pd.Series(centroid_latitude, name='centroid_lat')
    centroid_df = pd.DataFrame(columns=['centroid_lon', 'centroid_lat'])
    centroid_df['centroid_lon'] = centroid_longitude
    centroid_df['centroid_lat'] = centroid_latitude
    geometry_centroids = [Point(xy) for xy in zip(centroid_df['centroid_lon'], centroid_df['centroid_lat'])]
    geo_data_centroids = gpd.GeoDataFrame(centroid_df, geometry=geometry_centroids, crs="EPSG:4326")

    geometry = [Point(xy) for xy in zip(df['Lon'], df['Lat'])]
    geo_data = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    nyc_map = gpd.read_file(gpd.datasets.get_path('nybb'), crs="EPSG:4326")
    nyc_map = nyc_map.to_crs(epsg=4326)

    ax = nyc_map.plot(figsize=(10, 10), color='white', edgecolor='black')
    geo_data[geo_data['cluster'] == 0].plot(ax=ax, markersize=1, alpha=1, color='blue')
    geo_data[geo_data['cluster'] == 1].plot(ax=ax, markersize=1, alpha=1, color='green')
    geo_data[geo_data['cluster'] == 2].plot(ax=ax, markersize=1, alpha=1, color='orange')
    geo_data[geo_data['cluster'] == 3].plot(ax=ax, markersize=1, alpha=1, color='yellow')
    geo_data[geo_data['cluster'] == 4].plot(ax=ax, markersize=1, alpha=1, color='purple')
    geo_data[geo_data['cluster'] == 5].plot(ax=ax, markersize=1, alpha=1, color='cyan')
    geo_data[geo_data['cluster'] == 6].plot(ax=ax, markersize=1, alpha=1, color='olive')
    geo_data[geo_data['cluster'] == 7].plot(ax=ax, markersize=1, alpha=1, color='gray')
    geo_data[geo_data['cluster'] == 8].plot(ax=ax, markersize=1, alpha=1, color='pink')
    geo_data[geo_data['cluster'] == 9].plot(ax=ax, markersize=1, alpha=1, color='brown')
    geo_data_centroids.plot(ax=ax, markersize=50, alpha=1, color='red', marker='x')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Pickup location cluster centers")
    plt.savefig('Clusters on the map of New York City')
