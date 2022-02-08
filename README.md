# LSTM-for-Small-Area-Populations

Forecasting Small Area Population Totals with Long short-term memory models

Prerequisites
**Please note that these forecasts were run on the GPU-enabled version of TensorFlow.  Please ensure your environment is set up to allow TensorFlow to run on GPU.
A virtual anaconda environment was used. I have provided both a spec list which you can use to recreate the environment if you are using windows, and a ymp file (EP2.yml) which you can use if you are using other platforms.  I used EP2 as the name as the code was run on an earlier version of Python (version 3.7.9 64-bit). Please see https://www.anaconda.com/blog/moving-conda-environments for details on how to use.

#####
In the ‘data’ subfolder you will find:
-	ERPs in the AUS_ERPS_1991to2016_withSummedRemainder file.  This file is called when the forecasts are run.
-	 ‘Statistical Area Level 2 2011 to Remoteness Area 2011 concordances’ are provided in a csv file

#####
In the ‘src’ file’
Files to run the forecasts.
Several variants of the same script are provided: 
AUS_2006_30x5_500x2_P10_P50 – used for the 2006 based forecasts.
AUS_2011_30x_500x2_P10_P50 – used for the 2011 based forecasts.

Additionally, the script to produce the 2006-based Australian forecasts with a single stage of training and validation (as mentioned in the Methods section) is provided.
SingleTrain_AUS_2006_30x5_500x2_P10_P50

#####
In the ‘out’ folder are files generated when the forecasts are run.
There are three subfolders:
-	AUS_2006_30x5_500x2_P10_P50 – models, forecasts, and other information related to the 2006-based forecasts.

-	AUS_2011_30x_500x2_P10_P50 – models, forecasts, and other information related to the 2011-based forecasts.

-	SingleTrain_AUS_2006_30x5_500x2_P10_P50 – models, forecasts, and other information related to the 2006-based forecasts with a single round of training and validation.

In each of these folders you will find the following sub-folders
History – with details on the model training process, including the learning rate, training data loss, and validation data loss at every training epoch for each of the models.
Models – the trained models.  You can load these to replicate the forecasts.
Predictions – Includes error summaries and the forecasts made by each of the models.
Summary – summaries of the model architectures in txt files

Additionally, you will see 3 csv files
AllPredictions.csv – All forecasts, errors, and error summaries merged into 1 file.  
ErrorSummary.csv – MedAPE, MAPE, Percentage Bad Forecasts >10%, and Percentage Bad Forecasts >20%
Summary.csv – information about training, including the time to train a model and run the forecasts.  This was used to keep track of the experiments.
