import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from ucimlrepo import fetch_ucirepo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#Performance Calculations
def calc_performance(confusion_matrix):
    TP = confusion_matrix[0][0]
    FN = confusion_matrix[0][1]
    FP = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]

    P = TP + FN
    N = TN + FP

    TPR = TP/P
    TNR = TN/N
    FPR = FP/N
    FNR = FN/P

    Precision = TP/(TP + FP)
    F1_measure = (2*TP)/(2*TP + FP + FN)
    Accuracy = (TP + TN)/(P + N)
    Error_rate = (FP + FN)/(P + N)

    BACC = (TPR + TNR)/2
    TSS = (TP/(TP+FN)) - (FP/(FP + TN))
    HSS = (2 * (TP * TN - FP * FN))/((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) 
    
    return [TP, FN, FP, TN, TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS]

#Random Forest
def rf_classifier(features_train, features_test, targets_train, targets_test):
    rf = RandomForestClassifier()
    rf.fit(features_train, targets_train)

    rf_prediction = rf.predict(features_test)

    conf_matrix = metrics.confusion_matrix(targets_test, rf_prediction)
    performance = calc_performance(conf_matrix)

    #calculate additional metrics
    brier = metrics.brier_score_loss(targets_test, rf.predict_proba(features_test)[:, 1])
    roc_auc = metrics.roc_auc_score(targets_test, rf.predict_proba(features_test)[:, 1])
    performance.append(brier)
    performance.append(roc_auc)
    return performance

#KNN
def knn_classifier(features_train, features_test, targets_train, targets_test):

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features_train, targets_train)
    knn_prediction = knn.predict(features_test)
    
    conf_matrix = metrics.confusion_matrix(targets_test, knn_prediction)
    performance = calc_performance(conf_matrix)

    #calculate additional metrics
    brier = metrics.brier_score_loss(targets_test, knn.predict_proba(features_test)[:, 1])
    roc_auc = metrics.roc_auc_score(targets_test, knn.predict_proba(features_test)[:, 1])
    performance.append(brier)
    performance.append(roc_auc)
    return performance

#LSTM
def lstm_classifier(features_train, features_test, targets_train, targets_test):
    Xtrain, Xtest, ytrain, ytest = map(np.array, [features_train, features_test, targets_train, targets_test])
    shape = Xtrain.shape
    
    Xtrain_reshaped = Xtrain.reshape(len(Xtrain), shape[1], 1)
    Xtest_reshaped = Xtest.reshape(len(Xtest), shape[1], 1)

    lstm = Sequential()
    lstm.add(LSTM(10, activation='relu', input_shape=(57, 1), return_sequences=False))
    lstm.add(Dense(1, activation="sigmoid"))
    lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    lstm.fit(Xtrain_reshaped, ytrain, validation_data=(Xtest_reshaped, ytest), epochs=10, batch_size=64, verbose=0) 

    pred_prob = lstm.predict(Xtest_reshaped)
    y_pred = (pred_prob >= 0.5).astype(int)
    
    y_pred = y_pred.reshape(-1)
    ytest = ytest.reshape(-1)

    conf_matrix = metrics.confusion_matrix(ytest, y_pred)
    performance = calc_performance(conf_matrix)
    #calculate additional metrics
    brier = metrics.brier_score_loss(ytest, pred_prob)
    roc_auc = metrics.roc_auc_score(ytest, pred_prob)
    performance.append(brier)
    performance.append(roc_auc)

    return performance

# fetch dataset 
spambase = fetch_ucirepo(id=94) #94 for spam
  
# data (as pandas dataframes) 
features = spambase.data.features #X
targets = spambase.data.targets #y
targets = np.ravel(targets)

metric_names = ['TP', 'FN', 'FP', 'TN', 'TPR', 'TNR', 'FPR', 
                'FNR', 'Precision', 'F1_measure', 'Accuracy', 'Error_rate', 'BACC', 
                'TSS', 'HSS', 'Brier Score', 'AUC']

kf = KFold(n_splits=10, shuffle=True, random_state=1)

all_rf_metrics = []
all_knn_metrics = []
all_lstm_metrics = []

for i, (train_index, test_index) in enumerate(kf.split(features), start=1):
    #Split training and test data sets
    X_train, X_test, y_train, y_test = train_test_split(features, 
    targets, test_size=0.1, stratify=targets)

    #Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #Train models
    #Random Forest
    rf_performance = rf_classifier(X_train, X_test, y_train, y_test)
    #KNN
    knn_performance = knn_classifier(X_train, X_test, y_train, y_test)
    #LSTM
    lstm_perfomance = lstm_classifier(X_train, X_test, y_train, y_test)

    performance_metrics = pd.DataFrame([rf_performance, knn_performance, lstm_perfomance], columns=metric_names, index=['RF',
    'KNN', 'LSTM'])
    print("\n***Iteration {} Performance Metrics***" .format(i))
    print(performance_metrics)

    all_rf_metrics.append(rf_performance)
    all_knn_metrics.append(knn_performance)
    all_lstm_metrics.append(lstm_perfomance)

print("\n***Performance Summary for Individual Algorithms***\n")
metric_iter_names = ['iter1', 'iter2', 'iter3', 'iter4', 'iter5', 'iter6', 'iter7', 'iter8', 'iter9', 'iter10']

all_rf_metrics_df = pd.DataFrame(all_rf_metrics, columns=metric_names, index=metric_iter_names)
print("\n***All Metrics for All Interations: Random Forest***")
print(all_rf_metrics_df)

all_knn_metrics_df = pd.DataFrame(all_knn_metrics, columns=metric_names, index=metric_iter_names)
print("\n***All Metrics for All Interations: K Nearest Neighbors***")
print(all_knn_metrics_df)

all_lstm_metrics_df = pd.DataFrame(all_lstm_metrics, columns=metric_names, index=metric_iter_names)
print("\n***All Metrics for All Interations: LSTM***")
print(all_lstm_metrics_df)

print("\n***Average Performance of All Algorithms***\n")
avg_rf = all_rf_metrics_df.mean()
avg_knn = all_knn_metrics_df.mean()
avg_lstm = all_lstm_metrics_df.mean()

avg_all = pd.DataFrame({'RF': avg_rf, 'KNN': avg_knn, 'LSTM': avg_lstm}, index=metric_names)
print(avg_all)
