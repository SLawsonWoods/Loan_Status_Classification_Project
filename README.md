# Loan Approval - A classification project
Sarah Lawson Woods October 2021
________________________________________________________________________________________________________
**Table of Contents**

1.)  Project Summary
2.)  Project Objective & Goals
3.)  Audience
4.)  Project Deliverables


5.)  Project Context

6.)  Data Dictionary
7.)  Initial Hypothesis
8.)  Executive Summary - Conclusions & Next Steps**
9.)  Pipeline Stages Breakdown


10.)  Project Plan https://trello.com/b/OcXQ7aT0/loan-determination
10a.) Data Acquisition
10b.) Data Preparation
10c.) Data Exploration
10d.) Modeling and Evaluation

11.)  Reproduce

________________________________________________________________________________________________________
**Project Summary**
A Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. 
_________________________________________________________________________________________________________
**Project Objectives & Goals**
Document code, process (data acquistion, preparation, exploratory data analysis including  statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report. Create modules (acquire.py, prepare.py) that make your process repeateable. Construct at least 3 models to predict loan status of applicants. Prepare a final notebook ready for peer review with enough documentation to guide reviewer through the steps I took and why.  
________________________________________________________________________________________________________
**Audience**
The Codeup Data Science team.

___________________________________________________________________________________________
**Project Deliverables**

A Jupyter Notebook Report
A README.md
A .py file for aquiring and one for preparing the data
A project summary.

________________________________________________________________________________________________________
**Project Context**
My loan dataset comes from https://data-flair.training/blogs/machine-learning-project-ideas/

________________________________________________________________________________________________________
**Data Dictionary**




| Feature               | Datatype               | Description                                   |
|:----------------------|:-----------------------|:----------------------------------------------|
| Loan_ID               | 1311 non-null: int64   | Unique Loan ID                                |
| Gender                | 1311 non-null: object  | Male/ Female                                  |
| Married               | 1304 non-null: object  | Applicant married (Y/N)                       |
| Dependents            | 1310 non-null: object  | Number of dependents                          |
| Education             | 955 non-null: object   | Applicant Education (Graduate/ Under-Graduate)|
| Self_Employed         | 270 non-null: object   | Applicant income                              |
| ApplicantIncome       | 311 non-null: int64    | Mill where the beans were processed           |  |CoapplicantIncome      | 1311 non-null: int64   | Coapplicant income                            |
| Credit History        | 1165 non-null: object  | Credit history meets guidelines               |
| LoanAmount            | 1311 non-null: int64   | Loan amount in thousands                      |
| Loan_Amount_Term      | 1311 non-null: int64   | Term of loan in months                        |       
| Property_Area         | 1088 non-null: object  | Urban/ Semi Urban/ Rural                      |
| Loan_Status           | 1254 non-null: object  |   Loan approved (Y/N)                         |

______________________________________________________________________________________________________
**Initial Hypotheses**

My initial ideas were that there could be a strong relationship between applicant income and loan approval and that being male or female might have an impact on loan approval as well.
   
_________________________________________________________________________________________________________
**Executive Summary - Conclusions & Next Steps**

Goals: The purpose of this project is to create different classification models to most accurately predict loan approval status.

Target: Loan_Status, yes or no.

Findings: I created 6 models, a Decision Tree, KNN, 2 Logistic Regression models varying C value and two Random Forest Models with Model 2 performing the best at 84% accuracy on validate which beat my baseline predication of 69.4%.


Results: The best predictors I discovered in exploration and modeling were Credit History, Loan_Amount, Married and debatably gender.  In my visuals I found gender to have a higher correlation value than most of the features, I also did a chi squared test to evaluate if there was a relationship and found there was a significant relationship, but in feature importance gender was listed as one of the least important features.

Conclusion and Next Steps: 
-Some of the features I guessed would be valuable like applicant income and gender were not in a strong relationship with the target. 
-We find that one ML method, random forest Model 2 improves tthe accuracy of default predictions the most in the loan dataset. This ML method is superior to the baseline model of predicting the mode.

Next Steps: With more time I would create loops to create a multitude of models varying their parameters and features to find the best accuracy.  I would also run more tests of models with and without gender to see what impact this feature truly has on the accuracy of predictions.

_________________________________________________________________________________________________________
**Pipeline Stages Breakdown**

Plan
- Create README.md with data dictionary, project and business goals, come up with initial hypotheses.
- Acquire data from https://data-flair.training/blogs/machine-learning-project-ideas/ and create a function to automate this process. Save the function in an preparee.py file to import into the Final Report Notebook.
- Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process, store the function in a prepare.py module, and prepare data in Final Report Notebook by importing and using the function.
- Explore the dataset after preparation using visualizations, statistical tests to identify potential 
  drivers loan approval or denial.
- Establish a baseline accuracy and document well how you came to this model.
- Clearly define two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- Train four different ML models incorporate drivers.
- Evaluate models on train and validate datasets.
- Choose the model with that performs the best and evaluate that single model on the test dataset.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.
 
Plan -> Acquire
Store functions that are needed to acquire the loan data stored in two csv'; make sure the prepare.py module contains the necessary imports to run my code. The final function will return a pandas DataFrame. Import the acquire function from the prepare.py module and use it to acquire the data in the Final Report Notebook.
Plot distributions of individual variables.

Plan -> Acquire -> Prepare
Store functions needed to prepare the loan data; make sure the module contains the necessary imports to run the code. The final function should do the following: - Split the data into train/validate/test. - Handle any missing values. - Handle erroneous data and/or outliers that need addressing. - Encode variables as needed. Scale columns for modeling - Create any new features, if made for this project.
Import the prepare function from the prepare.py module and use it to prepare the data in the Final Report Notebook.

Plan -> Acquire -> Prepare -> Explore
Answer key questions, my hypotheses, and figure out the features that can be used in a classification model to best predict the target variable, churn.
Run at least 2 statistical tests in data exploration. Document my hypotheses, set an alpha before running the tests, and document the findings.
Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify features that are related to loan status (the target), identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.
Summarize my conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.

Plan -> Acquire -> Prepare -> Explore -> Model
Establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 3 different models. Document these steps well.Train (fit, transform, evaluate) multiple models. Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe. Remove variables that seem to give no insight.

Based on the evaluation of the models using the train and validate datasets, choose the best model to try with the test data, once.Test the final model on the out-of-sample data (the testing dataset), summarize the performance, interpret and document the results.

Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate
Compare evaluation metrics across all the models I train and select the ones I want to evaluate using my validate dataframe.

Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Conclusion
Summarize key findings and takeaways.

Clearly call out the questions and answers I am analyzing as well as offer insights and recommendations based on my findings.

_________________________________________________________________________________________________________
**Reproduce My Project**


 Read this README.md
 Download the 2 csv files I have provided, prepare.py, and final_report.ipynb files into your working     directory
 Run the final_notebook.ipynb
 
_________________________________________________________________________________________________________

