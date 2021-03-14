import os
import pickle
import numpy as np 
import pandas as pd 
from datetime import datetime
from sklearn.metrics import f1_score
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
np.random.seed(1234)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.compat.v1.set_random_seed(1234)

pd.options.mode.chained_assignment = None
logging.getLogger('tensorflow').disabled = True

from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_path = 'data/Hotel-A-train.csv'
test_path = 'data/Hotel-A-test.csv'
validation_path = 'data/Hotel-A-validation.csv'
encoder_dict_path = 'data/feature_encoding.pickle'
scalar_dict_path = 'data/feature_scaling.pickle'
dnn_weights = 'data/dnn.h5'
submission_path = 'data/submission.csv'

label_dict = {
        'Check-In': 1, 
        'Canceled': 2,
        'No-Show' : 3
            }
def label_encoding(df_cat, train):
    if train:
        if not os.path.exists(encoder_dict_path):
            encoder_dict = defaultdict(LabelEncoder)
            encoder = df_cat.apply(lambda x: encoder_dict[x.name].fit_transform(x))
            encoder.apply(lambda x: encoder_dict[x.name].inverse_transform(x))
            with open(encoder_dict_path, 'wb') as handle:
                pickle.dump(encoder_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(encoder_dict_path, 'rb') as handle:
        encoder_dict = pickle.load(handle)
    return df_cat.apply(lambda x: encoder_dict[x.name].transform(x))

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%m/%d/%Y")
    d2 = datetime.strptime(d2, "%m/%d/%Y")
    return int(abs((d2 - d1).days))

def checkin_checkout_gap(row):
    Expected_checkin = row['Expected_checkin']
    Expected_checkout = row['Expected_checkout']
    return days_between(Expected_checkin, Expected_checkout)

def checkin_booking_gap(row):
    Expected_checkin = row['Expected_checkin']
    Booking_date = row['Booking_date']
    return days_between(Expected_checkin, Booking_date)
    
def extract_data(csv_path, train=False):
    df = pd.read_csv(csv_path)

    df['checkin_checkout_gap'] = df.apply(checkin_checkout_gap, axis=1)
    df['checkin_booking_gap'] = df.apply(checkin_booking_gap, axis=1)

    df = df.drop(['Reservation-id', 'Expected_checkin', 'Expected_checkout', 'Booking_date'], axis = 1)

    output = df['Reservation_Status'].values
    del df['Reservation_Status']

    numerical_inputs = df[['Age', 'Adults', 'Children', 'Babies', 'Discount_Rate', 'Room_Rate', 'checkin_booking_gap', 'checkin_checkout_gap']].values
    df_cat = df.drop(['Age', 'Adults', 'Children', 'Babies', 'Discount_Rate', 'Room_Rate', 'checkin_booking_gap', 'checkin_checkout_gap'], axis = 1)

    categorical_inputs = label_encoding(df_cat, train)

    X = np.concatenate([numerical_inputs, categorical_inputs], axis=1)
    Y = np.array([label_dict[l] for l in output]) -  1

    if train:
        if not os.path.exists(scalar_dict_path):
            scalar = StandardScaler()
            scalar.fit(X)
            with open(scalar_dict_path, 'wb') as handle:
                pickle.dump(scalar, handle, protocol=pickle.HIGHEST_PROTOCOL)  

        class_weights = compute_class_weight('balanced',
                                            np.unique(Y),
                                            Y)
        class_weights = {i : class_weights[i] for i in range(len(set(Y)))}


    with open(scalar_dict_path, 'rb') as handle:
        scalar = pickle.load(handle)


    X = scalar.transform(X)
    X, Y = shuffle(X, Y)
    if train:
        return X, Y, class_weights
    return X, Y

def get_data():
    Xtrain, Ytrain, class_weights = extract_data(train_path, train=True)
    Xval, Yval = extract_data(validation_path)
    
    return Xtrain, Ytrain, Xval, Yval, class_weights

def get_test_data():
    df = pd.read_csv(test_path)

    df['checkin_checkout_gap'] = df.apply(checkin_checkout_gap, axis=1)
    df['checkin_booking_gap'] = df.apply(checkin_booking_gap, axis=1)
    df = df.drop(['Expected_checkin', 'Expected_checkout', 'Booking_date'], axis = 1)
    
    Reservation_id = df['Reservation-id'].values
    del df['Reservation-id']

    numerical_inputs = df[['Age', 'Adults', 'Children', 'Babies', 'Discount_Rate', 'Room_Rate', 'checkin_booking_gap', 'checkin_checkout_gap']].values
    df_cat = df.drop(['Age', 'Adults', 'Children', 'Babies', 'Discount_Rate', 'Room_Rate', 'checkin_booking_gap', 'checkin_checkout_gap'], axis = 1)

    categorical_inputs = label_encoding(df_cat, False)

    X = np.concatenate([numerical_inputs, categorical_inputs], axis=1)

    with open(scalar_dict_path, 'rb') as handle:
        scalar = pickle.load(handle)

    X = scalar.transform(X)
    return Reservation_id, X

def f1_m(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_m = 2 * (precision * recall) / (precision + recall)
    return f1_m

class DNNmodel(object):
    def __init__(self):
        Xtrain, Ytrain, Xval, Yval, class_weights = get_data()
        self.Xtrain = Xtrain
        self.Xval = Xval
        self.Ytrain  = Ytrain
        self.Yval = Yval
        self.class_weights = class_weights
        self.size_output = len(set(self.Ytrain))
        self.n_features = int(self.Xtrain.shape[1])
        print(" Shape of Xtrain : {}".format(self.Xtrain.shape))
        print(" No: of Classes : {}".format(self.size_output))
        print(" Class weights :\n{}".format(self.class_weights))

    def classifier(self):
        inputs = Input(shape=(self.n_features,), name='input_dnn')
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.size_output, activation='softmax')(x)
        self.model = Model(inputs, outputs)

    def train(self):
        self.classifier()
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(0.001),
            metrics=[f1_m],
        )
        self.history = self.model.fit(
                            self.Xtrain,
                            self.Ytrain,
                            batch_size=64,
                            epochs=30,
                            validation_data=[self.Xval, self.Yval],
                            class_weight=self.class_weights
                            )

    def save_model(self):
        self.model.save(dnn_weights)

    def calc_f1(self, X, Y):
        P = self.model.predict(X)
        P = P.argmax(axis=-1)
        return f1_score(Y, P, average='macro')

    def load_model(self):
        loaded_model = load_model(dnn_weights)
        loaded_model.compile(
                        loss='sparse_categorical_crossentropy',
                        optimizer=Adam(0.001),
                        metrics=[f1_m]
                        )
        self.model = loaded_model

    def evaluations(self):
        print(self.calc_f1(self.Xtrain, self.Ytrain))
        print(self.calc_f1(self.Xval, self.Yval))

    def predictions(self):
        Reservation_id, X = get_test_data()
        print(X.shape)
        P = self.model.predict(X)
        Ypred = P.argmax(axis=-1)
        Ypred += 1
        print(set(Ypred))

        data_dict = { 
            'Reservation-id' : Reservation_id, 
            'Reservation_Status' : Ypred
        } 

        df = pd.DataFrame(data_dict) 
        df.to_csv(submission_path, index=False)


    def run(self):
        if os.path.exists(dnn_weights):
            self.load_model()
        else:
            self.train()
            self.save_model()
        self.evaluations()
        self.predictions()

m = DNNmodel()
m.run()