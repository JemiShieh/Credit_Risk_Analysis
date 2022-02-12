# Credit Risk Analysis

## Credit Risk Analysis Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. This analysis therefore employs different techniques to train and evaluate models with unbalanced classes. Use imbalanced-learn and scikit-learn libraries to build and evaluate machine learning models using resampling.
Analyze the credit card credit dataset from LendingClub, a peer-to-peer lending services company. Oversample the data using the `RandomOverSampler` and `SMOTE` algorithms, and undersample the data using the `ClusterCentroids` algorithm. Then use a combinatorial approach of over- and undersampling using the `SMOTEENN` algorithm. Next, compare two new ensemble machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk. Evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Credit Risk Analysis Results
Evaluation of the six machine learning credit risk models produced the following results:
* `RandomOverSampler` Oversampling Model
   - Balanced Accuracy Score:	0.6573
   - Precision Score: 		high_risk      0.01    
   				low_risk       1.00    
   - Recall Score:               		high_risk      0.71   
   				low_risk       0.60
	
[image]

* ` `SMOTE` Oversampling Model
   - Balanced Accuracy Score:	0.6622
   - Precision Score: 		high_risk      0.01    
   				low_risk       1.00    
   - Recall Score:               		high_risk      0.63   
   				low_risk       0.69
[image]

* `ClusterCentroids` Undersampling Model
   - Balanced Accuracy Score:	0.5443
   - Precision Score: 		high_risk      0.01    
   				low_risk       1.00    
   - Recall Score:               		high_risk      0.69   
   				low_risk       0.40
[image]

* `SMOTEENN` Over/Undersampling Model
   - Balanced Accuracy Score:	0.6447
   - Precision Score: 		high_risk      0.01    
   				low_risk       1.00    
   - Recall Score:               		high_risk      0.72 
   				low_risk       0.57
[image]

* `BalancedRandomForestClassifier` Ensemble Model
   - Balanced Accuracy Score:	0.7885
   - Precision Score: 		high_risk      0.03    
   				low_risk       1.00    
   - Recall Score:               		high_risk      0.70 
   				low_risk       0.87
[image]

* `EasyEnsembleClassifier` Ensemble Model
   - Balanced Accuracy Score:	0.9317
   - Precision Score: 		high_risk      0.09
   				low_risk       1.00    
   - Recall Score:               		high_risk      0.92 
   				low_risk       0.94
[image]


## Credit Risk Analysis Summary
The results of the analysis show the ensemble models outperforming the resampling models across all three metrics, with the recommended `EasyEnsembleClassifier` ensemble model performing best and the `ClusterCentroids` undersampling model performing worst.

However, despite the impressive high recall scores of the ensemble models, particularly the `EasyEnsembleClassifier`, their precision scores for predicting high risk credits are still under 0.10, indicating far too many false positives which would potentially limit lending opportunities to otherwise low risk credits.
