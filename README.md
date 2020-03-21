# Module17

## Project Overview 
Goal is to use JavaScript and the D3.js library to retrieve the coordinates and magnitudes of the earthquakes from the GeoJSON data. I used the Leaflet library to plot the data on a Mapbox map through an API request and create interactivity for the earthquake data.

This module has taught me the following:

1. Explain how a machine learning algorithm is used in data analytics.
2. Create training and test groups from a given data set.
3. Implement the logistic regression, decision tree, random forest, and support vector machine algorithms.
4. Interpret the results of the logistic regression, decision tree, random forest, and support vector machine algorithms.
5. Compare the advantages and disadvantages of each supervised learning algorithm.
6. Determine which supervised learning algorithm is best used for a given data set or scenario.
7. Use ensemble and resampling techniques to improve model performance.

        
## Documents

- credit_risk_ensemble_final.ipynb : challenge
- credit_risk_resampling_final.ipynb : challenge
- Resources
      - LoanStats_2019Q1.csv : dataset

## Summary

Using machine learning techniques to identfy credit risk - different models to predict data. 

# Module 13 Challenge 

## Challenge overview

1. Implement machine learning models
2. Use resampling to attempt to address class imbalance. 
3. Evaluate the performance of machine learning models.

## Summary/ Analysis and Recommendations

### credit_risk_resampling:
### Naive Random Over Sampling
balanced accuracy score:
For the Naive Random Oversampling model, the balanced accuracy score is ~66%. It detects ~66% of risks correctly - true positives and true negatives.

high-risk - racall and precision:
The high-risk has a recall score of .74/74%, which means the predicted true positives (high-risk) is only ~74%, and the other 26% that are actually high-risk were predicted as low-risk. The high-risk has a precision score of 0.01/1%. Out of all the predicted high-riskers, about 1% are actually high risk. The 99% of the predicted high-riskers are actually low-risk.

low-risk-recall and precision:
The low-risk has a recall score of .58/58%, which means the predicted true positives (low-risk) is only ~58%, and the other 42% that are actually low-risk were predicted as high-risk. The low-risk has a precision score of 1.00/100%. All predicted as low-risk are actually all low-risk, and predicted accurately.

quick summary:
Overall, the company tends to play it safer and are more sensitive to identify high-risk loans. Of all of high-risks identified/predicted, most of them (99%) are actually not high-risk but low risks. Of all the low-risks identified/predicted, most of them are accurate and are low-risk. The accuracy score of 66% is most likely lowered due to the predicted high-risk loans that are actually low-risk loans.

### SMOTEE Oversampling
balanced accuracy score:
For the SMOTEE Oversampling model, the balanced accuracy score is ~65%. It detects ~65% of risks correctly - true positives and true negatives.

high-risk - racall and precision:
The high-risk has a recall score of .62/62%, which means the predicted true positives (high-risk) is only ~62%, and the other 38% that are actually high-risk were predicted as low-risk. The high-risk has a precision score of 0.01/1%. Out of all the predicted high-riskers, about 1% are actually high risk. The 99% of the predicted high-riskers are actually low-risk.

low-risk-recall and precision:
The low-risk has a recall score of .68/68%, which means the predicted true positives (low-risk) is only ~68%, and the other 32% that are actually low-risk were predicted as high-risk. The low-risk has a precision score of 1.00/100%. All predicted as low-risk are actually all low-risk, and predicted accurately.

quick summary:
Overall, the company tends to play it safer and are more sensitive to identify high-risk loans. Of all of high-risks identified/predicted, most of them (99%) are actually not high-risk but low risks. Of all the low-risks identified/predicted, most of them are accurate and are low-risk. The accuracy score of 65% is most likely lowered due to the predicted high-risk loans that are actually low-risk loans.

### Undersampling
balanced accuracy score:
For the Undersampling model, the balanced accuracy score is ~64%. It detects ~64% of risks correctly - true positives and true negatives.

high-risk - racall and precision:
The high-risk has a recall score of .66/66%, which means the predicted true positives (high-risk) is only ~66%, and the other 34% that are actually high-risk were predicted as low-risk. The high-risk has a precision score of 0.01/1%. Out of all the predicted high-riskers, about 1% are actually high risk. The 99% of the predicted high-riskers are actually low-risk.

low-risk-recall and precision:
The low-risk has a recall score of .40/40%, which means the predicted true positives (low-risk) is only ~40%, and the other 60% that are actually low-risk were predicted as high-risk. The low-risk has a precision score of 1.00/100%. All predicted as low-risk are actually all low-risk, and predicted accurately.

quick summary:
Overall, the company tends to play it safer and are more sensitive to identify high-risk loans. Of all of high-risks identified/predicted, most of them (99%) are actually not high-risk but low risks. Of all the low-risks identified/predicted, most of them are accurate and are low-risk. The accuracy score of 64% is most likely lowered due to the predicted high-risk loans that are actually low-risk loans.

### Combination (Over and Under) Sampling
balanced accuracy score:
For the Niave Random Oversampling model, the balanced accuracy score is ~53%. It detects ~53% of risks correctly - true positives and true negatives.

high-risk - racall and precision:
The high-risk has a recall score of .72/72%, which means the predicted true positives (high-risk) is only ~72%, and the other 28% that are actually high-risk were predicted as low-risk. The high-risk has a precision score of 0.01/1%. Out of all the predicted high-riskers, about 1% are actually high risk. The 99% of the predicted high-riskers are actually low-risk.

low-risk-recall and precision:
The low-risk has a recall score of .57/57%, which means the predicted true positives (low-risk) is only ~57%, and the other 43% that are actually low-risk were predicted as high-risk. The low-risk has a precision score of 1.00/100%. All predicted as low-risk are actually all low-risk, and predicted accurately.

quick summary:
Overall, the company tends to play it safer and are more sensitive to identify high-risk loans. Of all of high-risks identified/predicted, most of them (99%) are actually not high-risk but low risks. Of all the low-risks identified/predicted, most of them are accurate and are low-risk. The accuracy score of 53% is most likely lowered due to the predicted high-risk loans that are actually low-risk loans.

### Recommendation
Out of all these models,I do not think there is one especially better than the others. All these models are quite similar in values.

If I had to recommend one model, I would recommend the Naive Random Oversampling model. The reason for this is because the it seems like the banks/loaning companies prefer to be safe and identify high-risk loans in order to prevents losing money. If this is the case, the high-risk recall score would be very beneficial, and the Naive Random Sample has the highest high-risk recall score of 0.74. This would mean we correctly identified/predicted more of the high risk loans out of all the actual high risk loans in the data.

### credit_risk_ensemble:
### Balanced Random Forest Classifier
balanced accuracy score:
For the Balanced Random Forest Classifier, the balanced accuracy score is ~79%. It detects ~79% of risks correctly - true positives and true negatives.

high-risk - racall and precision:
The high-risk has a recall score of .67/67%, which means the predicted true positives (high-risk) is only ~67%, and the other 33% that are actually high-risk were predicted as low-risk. The high-risk has a precision score of 0.04/4%. Out of all the predicted high-riskers, about 4% are actually high risk. The 96% of the predicted high-riskers are actually low-risk.

low-risk-recall and precision:
The low-risk has a recall score of .90/90%, which means the predicted true positives (predicted low-risk) is only ~90% true, and the other 10% that are actually low-risk were predicted as high-risk. The low-risk has a precision score of 1.00/100%. All predicted as low-risk are actually all low-risk, and predicted accurately.

quick summary:
Overall, the company tends to play it safer and are more sensitive to identify high-risk loans. Of all of high-risks predicted, 96% are actually low-risk, and 4% out of the predicted high-risk are correct (true-positives/ actual high-risk). Of all the low-risks predicted, most of them are accurate and are low-risk. The accuracy score of 79% is most likely lowered due to the predicted high-risk loans that are actually low-risk loans

### Easy Ensemble Adaboost
balanced accuracy score:
For the Easy Ensemble Adaboost, the balanced accuracy score is ~93%. It detects ~93% of risks correctly - true positives and true negatives.

high-risk - racall and precision:
The high-risk has a recall score of .92/92%, which means the predicted true positives (high-risk) is only ~92% true, and the other 8% that are actually high-risk were predicted as low-risk. The high-risk has a precision score of 0.09/9%. Out of all the predicted high-riskers, about 9% are actually high risk. The 91% of the predicted high-riskers are actually low-risk.

low-risk-recall and precision:
The low-risk has a recall score of .94/94% true, which means the predicted true positives (predicted low-risk) is only ~94%, and the other 6% that are actually low-risk were predicted as high-risk. The low-risk has a precision score of 1.00/100%. All predicted as low-risk are actually all low-risk, and predicted accurately.

quick summary:
Overall, the company tends to play it safer and are more sensitive to identify high-risk loans. Of all of high-risks predicted, 91% are actually low-risk, and 9% out of the predicted high-risk are correct (true-positives/ actual high-risk). Of all the low-risks predicted, most of them are accurate and are low-risk. The accuracy score of 93% is most likely lowered due to the predicted high-risk loans that are actually low-risk loans

### Recommendation
I would recommend using the Easy Ensemble Adaboost model. It has a higher accuracy of ~93% and identifying true positives.

Also, the precision value is higher, which means the modeling is better at predicting true low-risk and high-risk. This will allow to better predict actual high-risk loans, which will save the company money. You do not want to loan out money to people of high-risk because most likely you will not get that money returned with interest. If you loan out money to high-riskers, you will most likely lose money. The recall vaue is higher as well, which means more of the predicted high-risk and low risk are true/accurate.

