import numpy as np
import glob
import librosa
import argparse
import pandas as pd
import concurrent.futures

target_dict = {
    "male":0,
    "female":1
}

def get_track_features( path ):
    y, sr = librosa.load( path )
    mfcc = librosa.feature.mfcc( y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13 )
    spectral_center = librosa.feature.spectral_centroid( y=y, sr=sr, hop_length=HOP_LENGTH )
    chroma = librosa.feature.chroma_stft( y=y, sr=sr, hop_length=HOP_LENGTH)
    spectral_contrast = librosa.feature.spectral_contrast( y=y, sr=sr, hop_length=HOP_LENGTH )
    return mfcc, spectral_center, chroma, spectral_contrast

def extract_target( path ):
    file_name = path.split( "\\")[-1]
    print( "Extracting %s..." % file_name )
    class_name = file_name.split("-")[0]
    return class_name

def extract_track_feature( track, index ):
    print( "Extracting ", track )
    mfcc, spectral_center, chroma, spectral_contrast = get_track_features( track )
    feature_matrix = np.zeros((TIME_SERIES_LENGTH,MERGED_FEATURES_SIZE ))
    feature_matrix[:, 0:13] = mfcc.T[0:TIME_SERIES_LENGTH, :]
    feature_matrix[:, 13:14] = spectral_center.T[0:TIME_SERIES_LENGTH, :]
    feature_matrix[:, 14:26] = chroma.T[0:TIME_SERIES_LENGTH, :]
    feature_matrix[:, 26:33] = spectral_contrast.T[0:TIME_SERIES_LENGTH, :]
    return (feature_matrix, index)

def get_features_of_tracks( track_paths, gender ):
    data = np.zeros( (len(track_paths),TIME_SERIES_LENGTH,MERGED_FEATURES_SIZE ))
    classes = []
    futures = []
    with concurrent.futures.ProcessPoolExecutor( 10 ) as executor:
        for i,track in enumerate( track_paths ):
            classes.append( target_dict[gender[i]])
            future = executor.submit( extract_track_feature, track, i)
    for future in concurrent.futures.as_completed( futures ):
        data[future[1]] = future[0]

    return data, np.array(classes)


def get_tracks( path ):
    search_string = path + "/*.mp3"
    files = glob.glob( search_string, recursive=True )
    print( files )
    return files

def save_features( features, classes, name ):
    save_features_name = "features-" + name + ".npy"
    save_classes_name = "classes-" + name + ".npy"
    with open( save_features_name, "wb") as f:
        np.save( f, features )
    with open( save_classes_name, "wb") as f:
        np.save( f, classes )

def read_csv( path ):
    df = pd.read_csv( path, header=0, delimiter="," )
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--time_series_length", type=int, default=128)
    FLAGS, unknown = parser.parse_known_args()
    HOP_LENGTH = 128
    TIME_SERIES_LENGTH = FLAGS.time_series_length
    MERGED_FEATURES_SIZE = 33
    dataframe_train = read_csv("dataframe_gender_train.csv")
    dataframe_train = dataframe_train[dataframe_train["gender"]!="other"]
    dataframe_train.to_csv( "dataframe_gender_train.csv", columns=["filename","gender"])
    dataframe_test = read_csv("dataframe_gender_test.csv")
    dataframe_test = dataframe_test[dataframe_test["gender"]!="other"]
    dataframe_test.to_csv( "dataframe_gender_test.csv", columns=["filename","gender"])

    tracks_train = dataframe_train['filename'].tolist()
    labels_train = dataframe_train['gender'].tolist()
    train_features, train_labels = get_features_of_tracks( tracks_train, labels_train)
    save_features( train_features, train_labels, "train")

    tracks_test = dataframe_test['filename'].tolist()
    labels_test = dataframe_test['gender'].tolist()
    test_features, test_labels = get_features_of_tracks( tracks_test, labels_test)
    save_features( test_features, test_labels, "test")
