#In this script we evaluate two LSTM architectures, 
#a simple LSTM and a bidirectional LSTM model
#we evaluate several input window sizes and use the keras tuner
#for hyperparameter tuning
#Let's set the randomisation seeds and then import the required libraries

#%%
import random
random.seed(2021)
from numpy.random import seed
seed(2021)
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(2021)

import random as python_random
python_random.seed(2021)

#https://stackoverflow.com/questions/54473254/cudnnlstm-unknownerror-fail-to-find-the-dnn-implementation
#enable memory growth for 1 GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import IPython
from kerastuner.tuners import BayesianOptimization
from kerastuner import HyperModel
from numpy import array
from keras.models import Sequential, Model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional, Input
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import os
import time
from contextlib import redirect_stdout

os.environ['TF_DETERMINISTIC_OPS'] = '1'

#0 no scaling
#1 scaling on
SCALE_ON=1

#subfolder in the out folder to store outputs in.  Change for each run of forecasts
OutputsFolderName="AUS_2011_30x_500x2_P10_P50"

window_size=[5,8,11]

forecast_horizon = 5
NUM_FEATURES = 1
MaxTrials=30
EpochsTuning=5
EpochsTraining=500
ValidationSplit=0.2
PATIENCE1=10
PATIENCE2=50

REMOVE_THIS_MANY_ROWS_FROM_END=1 #to remove the 'Remainder' row from the 

#Let's get the data.  We will use the data without the remainder for training, but 
#we will include the remainder in the predictions. 
#Get the parent directory
parentDirectory=os.path.dirname(os.path.dirname(__file__))

#This is the path of the folder where we will store model summaries, training history, and predictions
outFolder=os.path.join(parentDirectory,"out",OutputsFolderName)
historydir=os.path.join(outFolder,'history')
modelsdir=os.path.join(outFolder,'models')
summarydir=os.path.join(outFolder,'summary')
predictionsdir=os.path.join(outFolder,'predictions')


if not os.path.exists(outFolder):
    os.makedirs(outFolder)
    os.makedirs(historydir)
    os.makedirs(modelsdir)
    os.makedirs(summarydir)
    os.makedirs(predictionsdir)


DataFile="AUS_ERPS_1991to2016_withSummedRemainder.csv"


data_file_path=os.path.join(parentDirectory,"data",DataFile)
dataset_train_full = pd.read_csv(data_file_path)

columns=dataset_train_full.columns

areaNames=dataset_train_full.iloc[:,2]

#The "Actual" ERPs in the forecast period.  This will be used to calculate forecast errors
ActualDataStart=columns.get_loc("2012_SA2")
ActualDataEnd=columns.get_loc("2016_SA2")
ActualTruth=dataset_train_full.iloc[:,ActualDataStart:(ActualDataEnd+1)]

#The Data up to the jump off year
from_Column_ERPs=columns.get_loc("1991_SA2")
to_Column_ERPs=columns.get_loc("2011_SA2")
dataset_for_forecasts=dataset_train_full.iloc[:,from_Column_ERPs:(to_Column_ERPs+1)]


y_full=dataset_for_forecasts.T #we transpose the array so each column is a small area
y_multiSeries=pd.DataFrame(y_full[:(len(y_full))])


#we don't want the remainder row in the training set
#but we must include in prediction.
y_multiSeries=y_multiSeries.iloc[:,:(y_multiSeries.shape[1]-REMOVE_THIS_MANY_ROWS_FROM_END)]
y_multiSeries=np.array(y_multiSeries)

OriginalDataForScaling = y_multiSeries.copy() #keep the original dataset so that we can use info from it to scale data


#####
#Next we define our functions to transform the data into the required format for training.

# Data transformation for single series
def lstm_data_transform(x_data, y_data, num_steps,forecast_horizon):
    """ Changes data to the format for LSTM training 
for sliding window approach """
#source https://towardsdatascience.com/how-to-reshape-data-and-do-regression-for-time-series-using-lstm-133dad96cd00
    # Prepare the list for the transformed data
    X, y = list(), list()
    # Loop through the entire data set
    for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps 
        # if index is larger than the size of the dataset, we stop
        if end_ix-1 >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]

        # Append the list with sequences
        X.append(seq_X)
        
    # Make final arrays
    x_array = np.array(X)

    return x_array

        
        
# Data transformation for a set of time series
def lstm_full_data_transform(x_data, y_data, num_steps,forecast_horizon,scale_on,OriginalDataForScaling):
    """ Changes data to the format for LSTM training 
    for sliding window approach.  We train across the full series """
    
    X, y = list(), list()

    if (SCALE_ON==1):
        x_data=(x_data-OriginalDataForScaling.min())/(OriginalDataForScaling.max()-OriginalDataForScaling.min())
        y_data=(y_data-OriginalDataForScaling.min())/(OriginalDataForScaling.max()-OriginalDataForScaling.min())

    # Loop of the entire data set
    for j in range(x_data.shape[1]):
        for i in range(x_data.shape[0]):
            # compute a new (sliding window) index
            end_ix = i + num_steps 
            # if index is larger than the size of the dataset, we stop
            if end_ix >= x_data.shape[0]:
                break

            # Get a sequence of data for x
            seq_X = x_data[i:end_ix,j]
            # Get only the last element of the sequency for y
            seq_y = y_data[end_ix,j]
            
            # Append the list with sequencies
            X.append(seq_X)
            y.append(seq_y)
    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)

    x_array=x_array.reshape(x_array.shape[0],x_array.shape[1],1)
    return x_array, y_array


# Data transformation for a set of time series.  Last nVal windows are set aside for validation data
def lstm_full_data_transform2(x_data, y_data, num_steps,forecast_horizon,scale_on,OriginalDataForScaling):
    """ Changes data to the format for LSTM training 
    for sliding window approach.  We train across the full series """
    X, y = list(), list()
    Xval, Yval=list(), list()
    Xtrain, Ytrain= list(), list()
    nVal=3

    if (SCALE_ON==1):
        x_data=(x_data-OriginalDataForScaling.min())/(OriginalDataForScaling.max()-OriginalDataForScaling.min())
        y_data=(y_data-OriginalDataForScaling.min())/(OriginalDataForScaling.max()-OriginalDataForScaling.min())
       
    # Loop of the entire data set
    for j in range(x_data.shape[1]):
        for i in range(x_data.shape[0]):
            # compute a new (sliding window) index
            end_ix = i + num_steps 
            # if index is larger than the size of the dataset, we stop
            if end_ix >= x_data.shape[0]:
                break

            # Get a sequence of data for x
            seq_X = x_data[i:end_ix,j]
            # Get only the last element of the sequency for y
            seq_y = y_data[end_ix,j]
            
            X.append(seq_X)
            y.append(seq_y)

        #keep the last nVal sequences for validation.  validation data will be used for early stopping
        Xval.append(X[-nVal:])
        Yval.append(y[-nVal:])

        Xtrain.append(X[:-nVal])
        Ytrain.append(y[:-nVal])
        X, y=list(), list()

    # Make final arrays
    x_array = np.array(Xtrain)
    y_array = np.array(Ytrain)

    x_array=x_array.reshape(x_array.shape[0]*x_array.shape[1],x_array.shape[2],1)
    y_array=y_array.reshape(y_array.shape[0]*y_array.shape[1])

    Xval = np.array(Xval)
    Yval = np.array(Yval)

    Xval=Xval.reshape(Xval.shape[0]*Xval.shape[1],Xval.shape[2],1)
    Yval=Yval.reshape(Yval.shape[0]*Yval.shape[1])

    return x_array, y_array,Xval,Yval

#Before doing the hyperparameter search a callback needs to be defined to clear the training outputs after each of the training steps
#source: https://www.tensorflow.org/tutorials/keras/keras_tuner
class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)


#Function to calculate Median Absolute Percentage Errors
def calcMAPEs(TheForecast,ActualTruth):
    #Lets calculate the MAPES
    MAPES=np.abs(np.array(TheForecast)-np.array(ActualTruth))/np.array(ActualTruth)*100

    #Let's calc the median of the MAPES
    medianMapes=np.median(MAPES,axis=0)
    meanMAPES=np.mean(MAPES,axis=0)
    MAPES_greater10=np.count_nonzero(MAPES > 10,axis=0)/(MAPES.shape[0])*100
    MAPES_greater20=np.count_nonzero(MAPES > 20,axis=0)/(MAPES.shape[0])*100

    ErrorSummary=pd.DataFrame([medianMapes,meanMAPES,MAPES_greater10,MAPES_greater20])
    ErrorSummary=ErrorSummary.rename(index={0:"medianMapes",
    1:"meanMAPES",
    2:" >10%",
    3:" >20%",
    })

    OneLine=pd.DataFrame(np.hstack((medianMapes,meanMAPES,MAPES_greater10,MAPES_greater20)).ravel())
    MAPES=pd.DataFrame(MAPES)
    return MAPES,ErrorSummary,OneLine

#put error values into the forecasts so when create summary array the error values are already there
def ArraysWithErrors(OneLine,ModelTypeName,num_steps_w,forecast_horizon,TheForecast,MAPES_pd,ErrorSummary,areaNames):
    
    ConcatArr=pd.concat([TheForecast.reset_index(drop=True),MAPES_pd.reset_index(drop=True),ErrorSummary.reset_index(drop=True)],axis=1)
    
    #let's add a line with details into our summary forecasts
    df=pd.DataFrame(columns=ConcatArr.columns)
    df.append(pd.Series(name='info'))
    df.loc[0,:]=ModelTypeName+" "+str(num_steps_w)+" steps"
    df2=pd.concat([df,ConcatArr])

    newIndex=pd.concat([pd.Series(["info"]),areaNames])
    ind=pd.DataFrame(newIndex)
    ind=ind.rename(columns={ind.columns[0]:"SA2"})
    df2['SA2_names']=ind['SA2']
    df2.set_index('SA2_names',inplace=True)

    #Let's add details to the error summary so that when we aggregate them we know
    #which one is which
    df_errorArray=pd.DataFrame(columns=ErrorSummary.columns)
    df_errorArray.append(pd.Series(name='info'))
    df_errorArray.loc[0,:]=ModelTypeName+" "+str(num_steps_w)+" steps"
    ErrorSummary=pd.concat([df_errorArray,ErrorSummary])
    ErrorSummary=ErrorSummary.rename(index={0:"info"})
    
    return ErrorSummary,df2



#Function to perform forecasts
def runForecasts(dataset_train_full,num_features,num_steps,forecast_horizon,filename1,model1,SCALE_ON,OriginalDataForScaling,outFolder,areaNames):
    for c_f in range(len(dataset_train_full)):
        if (c_f % 100 == 0):
            print(c_f)
        if c_f==0:
            #We will measure execution time for each loop
            ETime=pd.DataFrame(index=range(len(dataset_train_full)), columns=['eTime'])
        
        start1=time.time()


        columns=dataset_train_full.columns
        from_Column_ERPs=columns.get_loc("1991_SA2")
        to_Column_ERPs=columns.get_loc("2011_SA2")
        
        y_full=dataset_train_full.iloc[c_f,from_Column_ERPs:(to_Column_ERPs+1)]
        y=y_full.reset_index(drop=True)
        Area_name=areaNames.iloc[c_f]
        ActualData=pd.DataFrame(y_full)
        ActualData=ActualData.rename(columns={"0":Area_name})
        ActualData_df=ActualData.T        

        if (SCALE_ON==1):
            y=(y-OriginalDataForScaling.min())/(OriginalDataForScaling.max()-OriginalDataForScaling.min())

        y=np.array(y)

        x_new = lstm_data_transform(y, y, num_steps=num_steps, forecast_horizon= forecast_horizon)
        
        x_train = x_new

        test_input=x_new[-1:]
        test_input_prescaled=test_input
        temp1=test_input

        test_input=test_input.reshape(1,num_steps,1)

        PredictionsList=list()
        LSTMPredictionsList=list()

        #Do the rolling predictions 
        for i in range(forecast_horizon):             
            test_input=test_input.reshape(1,num_steps,1)
            test_input=np.asarray(test_input).astype('float32')
            test_output = model1.predict(test_input, verbose=0)
            
            test_input[0,:(num_steps-1)]=test_input[0,1:]
            LSTMPredictionsList.append(test_output.reshape(1))
    
            test_input[0,(num_steps-1)]=test_output
            PredictionsList.append(test_output.reshape(1))
                
        PredictionsList=np.array(PredictionsList)
        predictions=PredictionsList.reshape(forecast_horizon,num_features)
        
        if (SCALE_ON==1):
            df_predict=predictions*(OriginalDataForScaling.max()-OriginalDataForScaling.min())+OriginalDataForScaling.min()

        else:
            df_predict=predictions
        
        #Let's label the predictions for each of the areas in a larger dataframe
        df_predict=pd.DataFrame(df_predict)
        
        for count_through_columns in range(num_features):
            df_predict=df_predict.rename(columns={df_predict.columns[count_through_columns]:Area_name})

        temp_predict_df=df_predict.T        
        
        if c_f==0:
            full_array_SA2s=temp_predict_df.iloc[[0]]

        else:
            frames_SA2=[full_array_SA2s, temp_predict_df.iloc[[0]]]
            full_array_SA2s=pd.concat(frames_SA2)

        endLoop=time.time()
        ETime.iat[c_f,0]=endLoop-start1

    timestr = time.strftime("%Y%m%d")
    FullArrayLocation=filename1+timestr+".csv"

    parentDirectory=os.path.dirname(os.path.dirname(__file__))
    file_path_df=os.path.join(outFolder,'predictions',FullArrayLocation)

    full_array_SA2s.to_csv(file_path_df)

    #Return the predictions
    return full_array_SA2s


 #Class definition to enable variables to be passed into keras tuner for the "Simple" LSTM

class LSTMHyperModel(HyperModel):
    #reference: https://github.com/keras-team/keras-tuner/blob/master/examples/cifar10.py
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        x = inputs
        
        x = tf.keras.layers.LSTM(
            units=hp.Int('units',min_value=128,max_value=512,step=32),
            activation='relu', input_shape=self.input_shape)(x)
        x=tf.keras.layers.Dense(
            units=hp.Int('units',min_value=128,max_value=512,step=32),
            activation='relu')(x)
        x=tf.keras.layers.Dense(
            units=hp.Int('units',min_value=128,max_value=512,step=32),
            activation='relu')(x)        
        outputs = tf.keras.layers.Dense(1,activation='relu')(x)

        model = tf.keras.Model(inputs,outputs)

        model.compile(tf.keras.optimizers.Adam(learning_rate=0.001),loss='mse',metrics=['mse'])

        return model

       
class Bidirectional_LSTMHyperModel(HyperModel):
    #reference: https://github.com/keras-team/keras-tuner/blob/master/examples/cifar10.py
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        x = inputs
        
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=hp.Int('units',min_value=128,max_value=512,step=32),
                activation='relu', input_shape=self.input_shape))(x)
        x=tf.keras.layers.Dense(
            units=hp.Int('units',min_value=128,max_value=512,step=32),
            activation='relu')(x)
        x=tf.keras.layers.Dense(
            units=hp.Int('units',min_value=128,max_value=512,step=32),
            activation='relu')(x)
        outputs = tf.keras.layers.Dense(1,activation='relu')(x)

        model = tf.keras.Model(inputs,outputs)

        model.compile(tf.keras.optimizers.Adam(learning_rate=0.001),loss='mse',metrics=['mse'])

        return model
      



# we will test multiple window sizes to evaluate 
# what period of time in years is most helpful for
# small area population forecasts

#Let's keep a record of our experiments in our ExperimentsArray
ExperimentalInfo=list()

#Forecasts will be run for both the simple LSTM and the bidirectional LSTM models

#Flag to check if summary array has been defined
SummaryArrayFlag=0

for modelType in range(0,2):

    if modelType==0:
        #Then we run the forecasts with the simple LSTM model
        ModelTypeName="Simple_LSTM"
    else:
        ModelTypeName="Bidirectional"


    for current_window in range(len(window_size)):
        if current_window>0:
            del x_new_w,y_new_w, 
        
        #reset the seeds for every model
        seed(2021)
        tf.random.set_seed(2021)
        python_random.seed(2021)
        random.seed(2021)

        num_steps_w=window_size[current_window]
        INPUT_SHAPE=(num_steps_w, NUM_FEATURES)

        #we will record the time in seconds it takes for each
        #model to be tuned, trained, and then used for forecasts
        a=time.time()

        #let's do validation and get best number of epochs
        x_new_w, y_new_w,x_new_w_VAL,y_new_VAL=lstm_full_data_transform2(y_multiSeries, y_multiSeries, num_steps_w,forecast_horizon,SCALE_ON,OriginalDataForScaling)
            
        if modelType==0:
            #Then we run the forecasts with the simple LSTM model            
            LSTM_hypermodel=LSTMHyperModel(input_shape=INPUT_SHAPE)
            
        else:
            #Then we run the forecasts with the simple LSTM model
            LSTM_hypermodel=Bidirectional_LSTMHyperModel(input_shape=INPUT_SHAPE)
            
        projectName=ModelTypeName+"_"+str(num_steps_w)
        
        bayesian_opt_tuner = BayesianOptimization(
            LSTM_hypermodel,
            objective='mse',
            max_trials=MaxTrials,
            seed=2021,
            executions_per_trial=1,
            directory=os.path.normpath('C:/keras_tuning2'),
            project_name=projectName,
            overwrite=True)

        bayesian_opt_tuner.search(x_new_w, y_new_w,epochs=EpochsTuning,validation_data=(x_new_w_VAL,y_new_VAL),verbose=2,callbacks = [ClearTrainingOutput(),keras.callbacks.EarlyStopping(monitor='val_loss',  
            patience=PATIENCE1)])

        bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)

        #Get the best model
        Model1 = bayes_opt_model_best_model[0]
        
        #If validation error is not increasing after PATIENCE1 epochs the learning rate is reduced up to 5e-6
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                    patience=PATIENCE1, min_lr=0.000005, verbose=2)        
        
        history=Model1.fit(x_new_w,y_new_w,epochs=EpochsTraining,\
            validation_data=(x_new_w_VAL,y_new_VAL),callbacks = \
                [keras.callbacks.EarlyStopping(monitor='val_loss',  
        patience=PATIENCE2,mode='auto',restore_best_weights=True),reduce_lr], verbose=2)

        hist_getBest = Model1.history.history['val_loss']
        n_epochs_best = np.argmin(hist_getBest)+1

        #let's split the validation data into validation set and test set

        x_new_w2, x_new_w_VAL2, y_new_w2, y_new_VAL2 = train_test_split(x_new_w_VAL, y_new_VAL,test_size=ValidationSplit, random_state=2021)

        #we allow the learning rate to be reduced even further for the second training run to 1e-6
        reduce_lr2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                    patience=PATIENCE1, min_lr=0.000001, verbose=2) 

        history2=Model1.fit(x_new_w2,y_new_w2,epochs=EpochsTraining,\
            validation_data=(x_new_w_VAL2,y_new_VAL2),callbacks = \
                [keras.callbacks.EarlyStopping(monitor='val_loss',  
        patience=PATIENCE2,mode='auto',restore_best_weights=True),reduce_lr2], verbose=2)

        outFolder=os.path.join(parentDirectory,"out",OutputsFolderName)
        historydir=os.path.join(outFolder,'history')
        modelsdir=os.path.join(outFolder,'models')
        predictionsdir=os.path.join(outFolder,'predictions')

        timestr1 = time.strftime("%Y%m%d")
        ModelSaveName2=ModelTypeName+"_"+str(num_steps_w)
        save_model_here=os.path.join(modelsdir,(ModelSaveName2))
        Model1.save(save_model_here)
        SaveHistoryHere=os.path.join(historydir,(ModelSaveName2+".csv"))
        SaveHistory2Here=os.path.join(historydir,(ModelSaveName2+"_History2.csv"))


        #Lets convert the dictionary type to a pd dataframe
        history_df = pd.DataFrame(history.history) 
        history2_def=pd.DataFrame(history2.history)
        
        with open(SaveHistoryHere, mode = 'w') as f:
            history_df.to_csv(f)

        with open(SaveHistory2Here, mode = 'w') as f:
            history2_def.to_csv(f)

        SaveSummaryHere=os.path.join(outFolder,"summary",(ModelSaveName2+".txt"))
        with open(SaveSummaryHere, 'w') as f:
            with redirect_stdout(f):
                Model1.summary()

         
        # Now let's run the forecasts
        TheForecast=runForecasts(dataset_for_forecasts,NUM_FEATURES,num_steps_w,forecast_horizon,ModelSaveName2,Model1,SCALE_ON,OriginalDataForScaling,outFolder,areaNames)

        #Let's get the error summaries.  OneLine will be used for a summary array
        MAPES,ErrorSummary,OneLine=calcMAPEs(TheForecast,ActualTruth)
        OneLine=OneLine.rename(columns={OneLine.columns[0]:ModelTypeName+"_"+str(num_steps_w)+"_steps_"+str(forecast_horizon)+"_year"})
        ErrorSummary,df2=ArraysWithErrors(OneLine,ModelTypeName,num_steps_w,\
            forecast_horizon,TheForecast,MAPES,ErrorSummary,areaNames)        
        
        #Save the error summary to the predictions folder
        ErrorSummary_df=pd.DataFrame(ErrorSummary.copy())
        ErrorSummaryFilePath=os.path.join(predictionsdir,(ModelSaveName2+"ErrorSummary.csv"))
        with open(ErrorSummaryFilePath, mode = 'w') as f:
            ErrorSummary_df.to_csv(f)

        if (SummaryArrayFlag==0):
            SummaryArrayFlag=1
            FullForecastArray=df2
            FullErrorArray=OneLine

        else:
            FullForecastArray=pd.concat([FullForecastArray,df2],axis=1)
            FullErrorArray=pd.concat([FullErrorArray,OneLine],axis=1)

        b=time.time()
        c=b-a
        The_Learning_rate=tf.keras.backend.eval(Model1.optimizer.lr)

        print('window length is: ' + str(num_steps_w) + ', the time to run is: ' + str(c))

        ModelConfig=Model1.optimizer.get_config()

        OurExperiments=[ModelTypeName,ModelSaveName2,forecast_horizon,num_steps_w,NUM_FEATURES,c,ModelSaveName2,outFolder,\
        MaxTrials,EpochsTuning,EpochsTraining,ValidationSplit,PATIENCE1,PATIENCE2,The_Learning_rate,\
            SCALE_ON,REMOVE_THIS_MANY_ROWS_FROM_END,n_epochs_best,np.array(ModelConfig)]
        ExperimentalInfo.append(OurExperiments)
        del Model1, LSTM_hypermodel
        keras.backend.clear_session()



#Let's write the list of lists to a CSV
import csv

ExperimentalHistorySaveName=os.path.join(outFolder,'Summary.csv')
with open(ExperimentalHistorySaveName, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(ExperimentalInfo)

FullPredictionsTogetherPath=os.path.join(outFolder,'AllPredictions.csv')
FullForecastArray.to_csv(FullPredictionsTogetherPath)

FullErrorArrayTogetherPath=os.path.join(outFolder,'ErrorSummary.csv')
FullErrorArray.to_csv(FullErrorArrayTogetherPath)

# %%
