import seaborn as sns
import warnings
import pandas as pd
import numpy as np
import julian
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tzwhere import tzwhere
from datetime import datetime
from astral import LocationInfo
from astral.sun import sunset,sunrise,dawn,dusk
import pytz
from datetime import date
timezone = pytz.timezone('UTC')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
warnings.simplefilter(action='ignore', category=FutureWarning)

from timezonefinder import TimezoneFinder
tf = TimezoneFinder()

COLUMNS_TO_DROP_AFTER_ENGINEERING = ['NWCG_REPORTING_AGENCY',"DISCOVERY_HOUR", 'DISCOVERY_DATE', 'CONT_DATE', 'DISCOVERY_TIME', 'CONT_TIME', 'DISCOVERY_DOY', 'CONT_DOY', 'FIPS_NAME','FIPS_CODE', 'COUNTY', 'STATE', 'Unnamed: 0', 'Shape','SOURCE_SYSTEM_TYPE', 'DAYS_TILL_CONT', 'DAY_DIFFERENCE', 'DAY_DIFFERENCE_BINARY', 'SOURCE_SYSTEM','NWCG_REPORTING_UNIT_NAME','NWCG_REPORTING_UNIT_ID','SOURCE_REPORTING_UNIT',
'SOURCE_REPORTING_UNIT_NAME','LOCAL_FIRE_REPORT_ID','LOCAL_INCIDENT_ID','FIRE_CODE','FIRE_NAME','CONT_HOUR',
'ICS_209_INCIDENT_NUMBER','ICS_209_NAME','MTBS_ID','Season',
'MTBS_FIRE_NAME','FIRE_SIZE','OWNER_CODE','OWNER_DESCR','COMPLEX_NAME','FIRE_YEAR',
                                     'FIRE_SIZE_CLASS','is_june_to_september']
OTHER_FEATURES = ['OBJECTID', 'FOD_ID', 'FPA_ID',"discovery_month",'classified_as']

TO_DUMMY_LST = ["FIRE_SIZE_CLASS","Season","discovery_month","classified_as"]

# Julian to date calculator
def julian_to_date(jd):
    if pd.notnull(jd):
        # The offset between Julian date and Gregorian date
        jd_offset = 2440587.5

        # Convert Julian date to days since the start of the Gregorian calendar
        julian_days = jd - jd_offset

        # Convert days to seconds
        seconds = julian_days * 86400.0

        # Calculate the date
        date = datetime(1970, 1, 1) + timedelta(seconds=seconds)

        return date
    return -1


# adding binary 4th of july feature
def add_july_4_binary_feature(df):
    """
    Function to add a binary feature indicating whether the date is within a week
    before or after July 4th directly to a pandas DataFrame.

    Parameters:
    - df: pandas DataFrame containing the date column.

    Returns:
    - None (DataFrame is modified in place).
    """
    # Convert the column to datetime
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'])

    # Define the date range for a week before and after July 4th
    july_4_date = pd.Timestamp(df['DISCOVERY_DATE'].dt.year.iloc[0], 7, 4)  # July 4th of the first year in the dataset
    week_before = july_4_date - pd.Timedelta(days=7)
    week_after = july_4_date + pd.Timedelta(days=7)

    # Add binary feature indicating whether the date is within the defined range
    df['is_week_before_or_after_july_4'] = (
            (df['DISCOVERY_DATE'] >= week_before) & (df['DISCOVERY_DATE'] <= week_after)).astype(int)

# adding the season feature
def add_season_feature(df):
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'])
    df['Season'] = df['DISCOVERY_DATE'].dt.month.apply(lambda x: ['Winter', 'Spring', 'Summer', 'Fall'][((x + 1) % 12) // 3])


# handling nulls in time-related features
def time_null_handler(df):
    df['DISCOVERY_TIME'] = df['DISCOVERY_TIME'].fillna("-1.0")
    df['DISCOVERY_TIME'] = df['DISCOVERY_TIME'].astype(str)
    df['CONT_TIME'] = df['CONT_TIME'].fillna("-1.0")
    df['CONT_TIME'] = df['CONT_TIME'].astype(str)

# Define the extract_hours function
def extract_hours(discovery_time):
    if discovery_time == "-1.0":
        return int(-1)
    if len(discovery_time) == 6:
        return int(discovery_time[:2])
    if len(discovery_time) == 5:
        return int(discovery_time[0])
    else:
        return int(0)


# Apply the extract_hours function to create the 'DISCOVERY_HOUR' feature
def converting_dates_to_hours(df):
    df['DISCOVERY_HOUR'] = df['DISCOVERY_TIME'].astype(str).apply(extract_hours)
    df['CONT_HOUR'] = df['CONT_TIME'].astype(str).apply(extract_hours)
    df['CONT_DATE'] = df['CONT_DATE'].apply(julian_to_date)
    # Convert 'CONT_DATE' column to datetime format
    df['CONT_DATE'] = pd.to_datetime(df['CONT_DATE'], errors='coerce')

    # Extract date only
    df['CONT_DATE'] = df['CONT_DATE'].dt.date

    df['CONT_DATE'] = df['CONT_DATE'].fillna(-1.0)

    
    
# calculating the hours to containments of fire feature
def calc_hours_to_contain(df):
    # Convert date columns to datetime format
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'], errors='coerce')
    df['CONT_DATE'] = pd.to_datetime(df['CONT_DATE'], errors='coerce')

    # Convert DISCOVERY_TIME and CONT_TIME to timedelta
    df['DISCOVERY_TIME'] = pd.to_numeric(df['DISCOVERY_TIME'], errors='coerce', downcast='float')
    df['DISCOVERY_TIME'] = pd.to_timedelta(df['DISCOVERY_TIME'], unit='m', errors='coerce')
    df['CONT_TIME'] = pd.to_numeric(df['CONT_TIME'], errors='coerce', downcast='float')
    df['CONT_TIME'] = pd.to_timedelta(df['CONT_TIME'], unit='m', errors='coerce')

    df['DAY_DIFFERENCE'] = df['CONT_DOY'] - df['DISCOVERY_DOY']
    df['DAY_DIFFERENCE'] = df['DAY_DIFFERENCE'].apply(lambda x: x if x >= 0 else 0)
    df['DAY_DIFFERENCE_BINARY'] = df['DAY_DIFFERENCE'].apply(lambda x: 1 if x > 0 else 0)
    df['HOURS_TO_CONTAIN'] = (24*df['DAY_DIFFERENCE'] + 24*df['DAY_DIFFERENCE_BINARY']-df['DISCOVERY_HOUR'] + df['CONT_HOUR']).astype(int)

    twenty_fours = np.full_like(df['DISCOVERY_HOUR'], 24)

    df['HOURS_TO_CONTAIN'] = df['HOURS_TO_CONTAIN'].apply(lambda x: x + 24 if x < 0 else x)

# binary function to return if hour is between dawn and dusk
def get_is_between_dawn_dusk(row):
    state = row['STATE']
    latitude = row['LATITUDE']
    longitude = row['LONGITUDE']
    fire_date = row['DISCOVERY_DATE']
    hour = row['DISCOVERY_HOUR']
    if hour < 0:
        return 1
    try:
        timezone_str = tf.timezone_at(lng=longitude, lat=latitude)
        loc = LocationInfo(name='', region=state, timezone=timezone_str, latitude=latitude, longitude=longitude)
        today = datetime(fire_date.year, fire_date.month, fire_date.day)
        sunrise_time = dawn(loc.observer, date=today, tzinfo=loc.timezone)
        sunset_time = dusk(loc.observer, date=today, tzinfo=loc.timezone)
        int_sunrise =sunrise_time.hour 
        int_sunset =sunset_time.hour
        return int(int_sunrise + 1<=hour<=int_sunset - 1)
    except ValueError:
        return 0

# adding nightqday feature
def add_night_or_day_col(df):
    df["IS_DAY"] = df.apply(lambda row: get_is_between_dawn_dusk(row), axis=1)
    return df

# generating dummies for fire size
def dummies_fire_size_for_test(df):
    fire_class_lst = df['FIRE_SIZE_CLASS'].unique().tolist()
    for fire_size in fire_class_lst:
        cmp = "FIRE_SIZE_CLASS_" + fire_size
        df[cmp] = np.zeros(len(df))

    for index, row in df.iterrows():
        for fire_size in fire_class_lst:
            if row['FIRE_SIZE_CLASS'] == fire_size:
                df.at[index, "FIRE_SIZE_CLASS_" + fire_size] = 1
    return df

# calculating day until containment 
def calc_days_till_cont(df):
    # Convert date and time columns to datetime format
    df['CONT_DATE'] = pd.to_datetime(df['CONT_DATE'], errors='coerce')
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'])

    # Calculate the time difference in days
    df['DAYS_TILL_CONT'] = (df['CONT_DATE'] - df['DISCOVERY_DATE']).dt.days

    # Replace NaN values with -1 (if needed)
    df['DAYS_TILL_CONT'].fillna(-1, inplace=True)

# create bins with hour fire was discovered 
def create_discovered_hour_bins(df):
    """
    Create binary features based on time bins for the DISCOVERY_HOUR column.

    Parameters:
    data (DataFrame): The DataFrame containing the DISCOVERY_HOUR column.

    Returns:
    DataFrame: The DataFrame with binary features representing time bins.
    """
    # Define time bins
    time_bins = [(10, 16), (16, 22), (22, 4), (4, 10)]

    # Create binary features
    for start_hour, end_hour in time_bins:
        bin_name = f'{start_hour}-{end_hour}'
        df['DISCOVERY_HOUR_BIN_' + bin_name] = ((df['DISCOVERY_HOUR'] >= start_hour) & (df['DISCOVERY_HOUR'] < end_hour)).astype(int)
        if start_hour == 22 and end_hour == 4:
            df['DISCOVERY_HOUR_BIN_' + bin_name] = ((df['DISCOVERY_HOUR'] >= start_hour) | (df['DISCOVERY_HOUR'] < end_hour)).astype(int)
    return df


# adding is_weekend and is_holiday features
def add_time_events_features(df):
    """
    Add binary features is_weekend and is_holiday to the dataset.

    Parameters:
    df (DataFrame): The DataFrame containing a 'DISCOVERY_DATE' column.

    Returns:
    DataFrame: The DataFrame with added features.

    Example:
    df = add_time_features(df)
    """
    # Convert 'DISCOVERY_DATE' column to datetime
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'])

    # Add binary feature is_weekend
    df['is_weekend'] = (df['DISCOVERY_DATE'].dt.dayofweek >= 5).astype(int)

    # Initialize US federal holiday calendar
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df['DISCOVERY_DATE'].min(), end=df['DISCOVERY_DATE'].max())

    # Add binary feature is_holiday
    df['is_holiday'] = df['DISCOVERY_DATE'].isin(holidays).astype(int)
    df['discovery_month'] = df['DISCOVERY_DATE'].dt.month
    # Add dummy variable for months between June and September
    df['is_june_to_september'] = np.where((df['discovery_month'] >= 6) & (df['discovery_month'] <= 9), 1, 0)
    return df


# knn hyperparams fun
def knn_hyperparameter_tuning_with_validation(train_set, validation_set):
    train_features = train_set[['LONGITUDE', 'LATITUDE']]
    train_target = train_set['STAT_CAUSE_DESCR']
    validation_features = validation_set[['LONGITUDE', 'LATITUDE']]
    validation_target = validation_set['STAT_CAUSE_DESCR']
    best_k = None
    best_model = None
    best_auc_roc_score = 0
    
    for k in range(1, 16):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(train_features, train_target)
        y_pred_proba = knn_model.predict_proba(validation_features)
        auc_roc_score = roc_auc_score(validation_target, y_pred_proba, multi_class='ovr')
        if auc_roc_score > best_auc_roc_score:
            best_auc_roc_score = auc_roc_score
            best_k = k
            best_model = knn_model
    return best_model, best_k, best_auc_roc_score


# adding predictions 
def add_predictions_to_dataset(model, dataset):
    # Extract features from the dataset
    features = dataset[['LONGITUDE', 'LATITUDE']]
    
    # Predict on the dataset
    predictions = model.predict(features)
    
    # Add the predictions to the dataset as a new column
    dataset['classified_as'] = predictions    
    return dataset

# dummies fun
def dummies_of_feature(df,feature):
    dummies = pd.get_dummies(df[feature], prefix=feature)
    return pd.concat([df, dummies], axis=1)

# more dummies fun
def make_dummies_from_features_list(df,features):
    dummied_df = df
    for feature in features:
        print(feature)
        dummied_df = dummies_of_feature(dummied_df,feature)
    return dummied_df

def remove_features(df):
    df.drop(columns=COLUMNS_TO_DROP_AFTER_ENGINEERING + OTHER_FEATURES , inplace=True)
    return df

############################### PreProcess Function#########################################
def preprocess_data(data, test_data, validation_data,for_model=False):
    print("Starting preprocessing")
    
    # Time handling
    print("Converting Julian Time")
    data['DISCOVERY_DATE'] = data['DISCOVERY_DATE'].apply(julian_to_date)
    test_data['DISCOVERY_DATE'] = test_data['DISCOVERY_DATE'].apply(julian_to_date)
    validation_data['DISCOVERY_DATE'] = validation_data['DISCOVERY_DATE'].apply(julian_to_date)
    
    # july 4th feature handling 
    print("Adding 4th of July feature")
    add_july_4_binary_feature(data)
    add_july_4_binary_feature(test_data)
    add_july_4_binary_feature(validation_data)
    
    # adding season ad ['winter','summer',...]
    print("Adding season feature")
    add_season_feature(data)
    add_season_feature(test_data)
    add_season_feature(validation_data)
    
    # fill nas time
    print("Handling nulls")
    time_null_handler(data)
    time_null_handler(test_data)
    time_null_handler(validation_data)
    
    # converting str date vars to date format
    print("Converting dates to hours")
    converting_dates_to_hours(data)
    converting_dates_to_hours(test_data)
    converting_dates_to_hours(validation_data)

    # add hours_to_contain var 
    print("Adding hours to containment feature")
    calc_hours_to_contain(data)
    calc_hours_to_contain(test_data)
    calc_hours_to_contain(validation_data)
    
#     IS_DAY feature  
    print("Adding is_day feature")
    validation_data = add_night_or_day_col(validation_data)
    test_data = add_night_or_day_col(test_data)
    data = add_night_or_day_col(data)

    # days until cont feature
    print("Adding days_till_cont feature")
    calc_days_till_cont(data)
    calc_days_till_cont(test_data)
    calc_days_till_cont(validation_data)
    
    # Call the function and update the dataframe
    print("Create bins for discovered hour")
    data = create_discovered_hour_bins(data)
    test_data = create_discovered_hour_bins(test_data)
    validation_data = create_discovered_hour_bins(validation_data)

    # add holidays events feature 
    print("Add holiday feature")
    data = add_time_events_features(data)
    test_data = add_time_events_features(test_data)
    validation_data = add_time_events_features(validation_data)
    
    print("KNN classification starting...")
    #KNN classification for areas 
    knn_feature_model, best_k, auc_roc_score = knn_hyperparameter_tuning_with_validation(data, validation_data)
    add_predictions_to_dataset(knn_feature_model,data)
    add_predictions_to_dataset(knn_feature_model,test_data)
    add_predictions_to_dataset(knn_feature_model,validation_data)
    
    # make dummies of classified vars
    print("Creating dummies for data")
    data = make_dummies_from_features_list(data,TO_DUMMY_LST)
    print("Creating dummies for test_data")
    test_data = make_dummies_from_features_list(test_data,TO_DUMMY_LST)
    print("Creating dummies for validation_data")
    validation_data = make_dummies_from_features_list(validation_data,TO_DUMMY_LST)
  
    if for_model:
        # remove features 
        print("Removing features for model")
        data = remove_features(data)
        test_data = remove_features(test_data)
        validation_data = remove_features(validation_data)
    return data, test_data, validation_data


############################################ Failed attempt at enriching using land cover data #################################3
# import pygeohydro as gh
# import pyproj

# def enrich_land_cover(df) : 
#     df['NLCD_land_cover_2016'] = np.NaN
#     # transforming geographic data from NAD83 (epsg:4269) to the datum used by NLCD - WGS84 (epsg:4326) 
#     src_crs = 'EPSG:4269'  # NAD83
#     dst_crs = 'EPSG:4326'
    
#     # Create a PyProj transformer object
#     transformer = pyproj.Transformer.from_crs(src_crs, dst_crs)
#     # Define chunk size
# chunk_size = 100  # Adjust this value based on your system's memory

# # Filter out the rows with null values in the 'NLCD_land_cover_2019' column
# df_null = df[df['NLCD_land_cover_2016'].isnull()]

# # Number of chunks
# num_chunks = len(df_null) // chunk_size + 1

# # Process each chunk
# for i in range(num_chunks):
#     start = i * chunk_size
#     end = (i + 1) * chunk_size
#     df_chunk = df_null.iloc[start:end]
    
#     # Transform coordinates and get NLCD land cover for each chunk
#     transformed_coordinates = [transformer.transform(lon, lat) for lon, lat in zip(df_chunk['LONGITUDE'], df_chunk['LATITUDE'])]
    
#     # Retry mechanism
#     max_retries = 30  # Maximum number of retries
#     delay = 60  # Delay in seconds between retries

#     for attempt in range(max_retries):
#         try:
#             # Update the 'NLCD_land_cover_2016' column of the original DataFrame directly
#             cover_2016 = gh.nlcd_bycoords(transformed_coordinates, years={"cover": [2016]})['cover_2016']
#             for j, val in zip(df_chunk.index, cover_2016):
#                 df.at[j, 'NLCD_land_cover_2016'] = val
#             break  # If the function executes successfully, break the loop
#         except ServiceUnavailableError:
#             if attempt < max_retries - 1:  # If this wasn't the last attempt
#                 time.sleep(delay)  # Wait for a while before retrying
#             else:
#                 raise  # If this was the last attempt, re-raise the exception