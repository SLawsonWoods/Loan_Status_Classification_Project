import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

### Function from Curriclulm

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    
    return train, validate, test


def prep_function(df):
    # set loan id as index
    df = df.set_index('Loan_ID')
    # rename column
    df = df.rename(columns={'ApplicantIncome': "Applicant_Income",'CoapplicantIncome':                         "Coapplicant_Income", 'LoanAmount':"Loan_Amount"})
    # drop nulls in all but loan status
    df = df.dropna()
    return df

def encode(train, validate, test):
    # Encoding categorical data
    # Encoding the Independent Variable
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    # I am going to loop through each col of the type object and encode
    for col in train.select_dtypes(include = 'object'):
        train[col]=labelencoder_X.fit_transform(train[col]) 
        validate[col]=labelencoder_X.fit_transform(validate[col]) 
        test[col]=labelencoder_X.fit_transform(test[col]) 
    return train, validate, test

def split_2(train, validate, test): 
    # here I split into my features and target
    y_train = train[['Loan_Status']]
    X_train = train.drop(columns= 'Loan_Status')
    y_validate = validate[['Loan_Status']]
    X_validate = validate.drop(columns= 'Loan_Status')
    y_test = test[['Loan_Status']]
    X_test = test.drop(columns= 'Loan_Status')
    return train, validate, test, y_train, X_train, y_validate, X_validate, y_test, X_test
    
def scaling(X_train, X_validate, X_test): 
    # time to scale features
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_validate = sc.transform(X_validate)
    X_test = sc.transform(X_test)
    return X_train, X_validate, X_test


