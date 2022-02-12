# Credit Risk Analysis

## Credit Risk Analysis Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. This analysis therefore employs different techniques to train and evaluate models with unbalanced classes. Use imbalanced-learn and scikit-learn libraries to build and evaluate machine learning models using resampling.
Analyze the credit card credit dataset from LendingClub, a peer-to-peer lending services company. Oversample the data using the `RandomOverSampler` and `SMOTE` algorithms, and undersample the data using the `ClusterCentroids` algorithm. Then use a combinatorial approach of over- and undersampling using the `SMOTEENN` algorithm. Next, compare two new ensemble machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk. Evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Credit Risk Analysis Results
Evaluation of the six machine learning credit risk models produced the following results:
* `RandomOverSampler` Oversampling Model
   - Balanced Accuracy Score:	0.6573
   - Precision Score: 		high_risk      0.01    low_risk       1.00    
   - Recall Score:              high_risk      0.71    low_risk       0.60
	
![Screenshot (66)](https://user-images.githubusercontent.com/92612370/153698599-ef13d1e8-3b6c-4239-baa0-7e187c04a4e6.png)


* `SMOTE` Oversampling Model
   - Balanced Accuracy Score:	0.6622
   - Precision Score: 		high_risk      0.01    low_risk       1.00    
   - Recall Score:              high_risk      0.63    low_risk       0.69

![Screenshot (67)](https://user-images.githubusercontent.com/92612370/153698613-d241c840-f19c-405a-bbc5-b25c96faaf17.png)


* `ClusterCentroids` Undersampling Model
   - Balanced Accuracy Score:	0.5443
   - Precision Score: 		high_risk      0.01    low_risk       1.00    
   - Recall Score:              high_risk      0.69    low_risk       0.40

![Screenshot (69)](https://user-images.githubusercontent.com/92612370/153698624-06eedc64-3151-44c7-a02c-db7db41816b8.png)


* `SMOTEENN` Over/Undersampling Model
   - Balanced Accuracy Score:	0.6447
   - Precision Score: 		high_risk      0.01   low_risk       1.00    
   - Recall Score:              high_risk      0.72   low_risk       0.57

![Screenshot (70)](https://user-images.githubusercontent.com/92612370/153698647-2443b45d-464a-423b-bd5b-19349568d098.png)


* `BalancedRandomForestClassifier` Ensemble Model
   - Balanced Accuracy Score:	0.7885
   - Precision Score: 		high_risk      0.03    low_risk       1.00    
   - Recall Score:     		high_risk      0.70    low_risk       0.87

![Screenshot (71)](https://user-images.githubusercontent.com/92612370/153698652-e087c9ce-4bcf-4080-a627-ed35085618ff.png)


* `EasyEnsembleClassifier` Ensemble Model
   - Balanced Accuracy Score:	0.9317
   - Precision Score: 		high_risk      0.09    low_risk       1.00    
   - Recall Score:    		high_risk      0.92    low_risk       0.94
				
![Screenshot (72)](https://user-images.githubusercontent.com/92612370/153698666-f463ae62-44dd-4e24-9877-871be624cc03.png)


## Credit Risk Analysis Summary
The results of the analysis show the ensemble models outperforming the resampling models across all three metrics, with the recommended `EasyEnsembleClassifier` ensemble model performing best and the `ClusterCentroids` undersampling model performing worst.

However, despite the impressive high recall scores of the ensemble models, particularly the `EasyEnsembleClassifier`, their precision scores for predicting high risk credits are still under 0.10, indicating far too many false positives which would potentially limit lending opportunities to otherwise low risk credits.
