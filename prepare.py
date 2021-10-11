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


def prep_loan_data_modeling(train, validate, test):
        
    '''
    This function take in the telco_churn data acquired by get_connection,
    Returns prepped df with target column turned to binary, columns dropped that were not needed,           missing values in total_charges handled by deleting those 11 rows, dropping duplicates, and changing     total_charges to numeric)
    '''
    
    
    encoded_columns=['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Credit_History',       'Property_Area','Loan_Status']
    
    #make dummy variables
    dummy_df = pd.get_dummies(train_encoded[encoded_columns], dummy_na=False, drop_first=[True, True])
    
    # put it all back together
    train_encoded = pd.concat([train_encoded, dummy_df], axis=1)
    
    # drop initial column since we have that information now
    train_encoded = train_encoded.drop(columns=encoded_columns)
    
    #make dummy variables
    dummy_df = pd.get_dummies(validate_encoded[encoded_columns], dummy_na=False, drop_first=[True, True])
    
    # put it all back together
    validate_encoded = pd.concat([validate_encoded, dummy_df], axis=1)
    
    # drop initial column since we have that information now
    validate_encoded = validate_encoded.drop(columns=encoded_columns)
    
    #make dummy variables
    dummy_df = pd.get_dummies(test_encoded[encoded_columns], dummy_na=False, drop_first=[True, True])
    
    # put it all back together
    test_encoded = pd.concat([test_encoded, dummy_df], axis=1)
    
#     # drop initial column since we have that information now
#     test_encoded = test_encoded.drop(columns=encoded_columns)
    
#     train_encoded.drop(columns='paperless_billing',inplace=True)
    
#     validate_encoded.drop(columns='paperless_billing',inplace=True)
    
#     test_encoded.drop(columns='paperless_billing',inplace=True)
    
    return train_encoded, validate_encoded, test_encoded
    

    




