import geopandas as gpd
from shapely import Point
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


def remove_pickups_outside_of_nyc(df):
    geometry = [Point(xy) for xy in zip(df['Lon'], df['Lat'])]
    raw_geo_data = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    nyc_map = gpd.read_file(gpd.datasets.get_path('nybb'), crs="EPSG:4326")
    nyc_map = nyc_map.to_crs(epsg=4326)

    filtered_data = gpd.sjoin(raw_geo_data, nyc_map, how='right')
    filtered_data = filtered_data.drop(columns=['index_left', 'BoroCode', 'BoroName', 'Shape_Leng', 'Shape_Area', 'geometry'])
    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data


def append_weekday_column(df):
    weekdays = []
    for dt in df["Date/Time"]:
        date_time_object = datetime.strptime(dt, '%m/%d/%Y %H:%M:%S')
        weekdays.append(date_time_object.weekday())
    df['Weekday'] = weekdays

    return df


def append_date_time_columns(df):
    months = []
    days = []
    years = []
    hours = []
    minutes = []
    seconds = []

    for dt in df['Date/Time']:
        date_time_object = datetime.strptime(dt, '%m/%d/%Y %H:%M:%S')
        months.append(date_time_object.month)
        days.append(date_time_object.day)
        years.append(date_time_object.year)
        hours.append(date_time_object.hour)
        minutes.append(date_time_object.minute)
        seconds.append(date_time_object.second)

    df['Month'] = months
    df['Day'] = days
    df['Hour'] = hours
    df['Minute'] = minutes
    df = df.drop(columns=['Date/Time'])

    return df


def append_weekend_column(df):
    weekend = []
    for day in df['Weekday']:
        if day <= 4:
            weekend.append(0)
        if day > 4:
            weekend.append(1)
    df['Weekend'] = weekend

    return df


def label_encode_column(df, column):
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])

    return df


def get_dataframes_with_amount_per_cluster(df, cluster):
    df = df[df['cluster'] == cluster]
    amount_values = df['Hour'].value_counts()
    df['Amount'] = df['Hour'].map(amount_values)
    df = df.reset_index(drop=True)

    return df
