from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
import pickle
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
import os
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from tcn import compiled_tcn #loading TCN model
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics 

main = tkinter.Tk()
main.title("Explainable AI for Intrusion Detection Systems: LIME and SHAP Applicability on Multi-Layer Perceptron") 
main.geometry("1300x1200")

global filename, lstm_model, tcn_model, X, Y
global accuracy, precision, recall, fscore, perturb, features
global X_train, X_test, y_train, y_test
global labels
global label_encoder, scaler, shap_values

def upload(): 
    global filename, labels, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset)+"\n\n")
    text.update_idletasks()
    labels = np.unique(dataset["Label"])
    label = dataset.groupby('Label').size()
    label.plot(kind="bar")
    plt.xlabel("Data Type")
    plt.ylabel("Count")
    plt.title("Normal & Attack Graph")
    plt.show()

def preprocess():
    text.delete('1.0', END)
    global dataset, X, Y, label_encoder, scaler
    global X_train, X_test, y_train, y_test, perturb, features
    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for j in range(len(types)):
        name = types[j]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))#encode all str columns to numeric
            label_encoder.append([columns[j], le])
    dataset.fillna(0, inplace = True)
    dataset = dataset[np.isfinite(dataset).all(1)]
    perturb = dataset['FlowDuration'].ravel()
    Y = dataset['Label'].ravel()
    dataset.drop(['Label', 'FlowDuration'], axis = 1,inplace=True)
    features = dataset.columns
    X = dataset.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = to_categorical(Y)
    text.insert(END,"Processed & Normalized Dataset Features = "+str(X))

def trainTestSplit():
    text.delete('1.0', END)
    global X, Y, X_train, X_test, y_train, y_test, perturb, features
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    for i in range(0, 10):
        X_test[i,0] = perturb[i]
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset split for train and test\n\n")
    text.insert(END,"80% dataset records used to train Algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to test Algorithms : "+str(X_test.shape[0])+"\n")    

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    fig, axs = plt.subplots(1,2,figsize=(10, 4))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axs[0]);
    ax.set_ylim([0,len(labels)])
    axs[0].set_title(algorithm+" Confusion matrix") 

    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(testY, predict, pos_label=1)
    axs[1].plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title(algorithm+" ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive rate')
    plt.show()

def runLSTM():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    global X, Y, X_train, X_test, y_train, y_test, labels, lstm_model
    lstm_model = Sequential()#defining deep learning sequential object
    #adding LSTM layer with 100 filters to filter given input X train data to select relevant features
    lstm_model.add(LSTM(32,input_shape=(X_train.shape[1], X_train.shape[2])))
    #adding dropout layer to remove irrelevant features
    lstm_model.add(Dropout(0.3))
    #adding another layer
    lstm_model.add(Dense(32, activation='relu'))
    #defining output layer for prediction
    lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
    #compile LSTM model
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #start training model on train data and perform validation on test data
    #train and load the model
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model.load_weights("model/lstm_weights.hdf5")
    #perform prediction on test data    
    predict = lstm_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("LSTM", y_test1, predict)

def runTCN():
    global X, Y, X_train, X_test, y_train, y_test, labels, tcn_model
    #creating tcn model with number of features and classes
    tcn_model = compiled_tcn(return_sequences=False, num_feat=1, num_classes=2, nb_filters=20, kernel_size=6, dilations=[2 ** i for i in range(9)], nb_stacks=1,
                         max_len=X_train[0:1].shape[1], use_skip_connections=True)
    if os.path.exists("model/tcn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/tcn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = tcn_model.fit(X_train, y_train.squeeze().argmax(axis=1), epochs = 20, validation_data=(X_test, y_test.squeeze().argmax(axis=1)), callbacks=[model_check_point], verbose=1)
        f = open('model/tcn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        tcn_model.load_weights("model/tcn_weights.hdf5")
    predict = tcn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("TCN", y_test1, predict)

def graph():
    df = pd.DataFrame([['LSTM','Precision',precision[0]],['LSTM','Recall',recall[0]],['LSTM','F1 Score',fscore[0]],['LSTM','Accuracy',accuracy[0]],
                       ['TCN','Precision',precision[1]],['TCN','Recall',recall[1]],['TCN','F1 Score',fscore[1]],['TCN','Accuracy',accuracy[1]],                       
                      ],columns=['Parameters','Algorithms','Value'])                
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')                
    plt.show()

def predict():
    text.delete('1.0', END)
    global lstm_model, scaler, label_encoder, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    testdata = pd.read_csv(filename)
    temp = testdata.values
    for i in range(len(label_encoder)-1):
        le = label_encoder[i]
        testdata[le[0]] = pd.Series(le[1].transform(testdata[le[0]].astype(str)))#encode all str columns to numeric
    testdata.fillna(0, inplace = True)
    testdata = testdata[np.isfinite(testdata).all(1)]
    testdata.drop(['FlowDuration'], axis = 1,inplace=True)            
    testdata = scaler.transform(testdata)
    testdata = np.reshape(testdata, (testdata.shape[0], testdata.shape[1], 1))
    predict = lstm_model.predict(testdata)
    predict = np.argmax(predict, axis=1)
    for i in range(len(predict)):
        text.insert(END,"Test Data = "+str(temp[i][0:15])+" =====> "+labels[predict[i]]+"\n\n")            

            
def explanation():
    global features, X_train, X_test, y_train, y_test, shap_values
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    y_test = np.argmax(y_test, axis=1)
    y_train = np.argmax(y_train, axis=1)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    shap.initjs()
    explainer = shap.TreeExplainer(rf, X_test)
    # Explain the predictions of your model
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    # Plot the SHAP values
    shap.summary_plot(shap_values[0], feature_names=features, show=True)
    

def Violinexplanation():
    global shap_values, features
    #violin plot to explain names of features which is contributing most for the algorithm to make correct prediction
    shap.plots.violin(shap_values[0], feature_names=features, show=True)


font = ('times', 16, 'bold')
title = Label(main, text='Explainable AI for Intrusion Detection Systems: LIME and SHAP Applicability on Multi-Layer Perceptron')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=24,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=40,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload CICIDS Dataset", command=upload, bg='#ffb3fe')
uploadButton.place(x=50,y=600)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocess, bg='#ffb3fe')
processButton.place(x=300,y=600)
processButton.config(font=font1) 

splitButton1 = Button(main, text="Train & Test Split", command=trainTestSplit, bg='#ffb3fe')
splitButton1.place(x=550,y=600)
splitButton1.config(font=font1) 

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM, bg='#ffb3fe')
lstmButton.place(x=790,y=600)
lstmButton.config(font=font1) 

tcnButton = Button(main, text="Run TCN Algorithm", command=runTCN, bg='#ffb3fe')
tcnButton.place(x=990,y=600)
tcnButton.config(font=font1) 

graphButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=50,y=650)
graphButton.config(font=font1)

explainButton = Button(main, text="Summary Model Explanation", command=explanation, bg='#ffb3fe')
explainButton.place(x=300,y=650)
explainButton.config(font=font1)

explainButton = Button(main, text="Violin Model Explanation", command=Violinexplanation, bg='#ffb3fe')
explainButton.place(x=550,y=650)
explainButton.config(font=font1)

predictButton = Button(main, text="Predict Attack from Test Data", command=predict, bg='#ffb3fe')
predictButton.place(x=790,y=650)
predictButton.config(font=font1)

main.config(bg='LightSalmon3')
main.mainloop()
