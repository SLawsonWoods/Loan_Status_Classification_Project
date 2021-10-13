import pandas as pd
import numpy as np
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
#from sklearn.tree import export_graphviz
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



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
    df = df.rename(columns={'ApplicantIncome': "Applicant_Income",'CoapplicantIncome':                         "Coapplicant_Income", 'LoanAmount':"Loan_Amount", 'Gender':'Is_male'})
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
    list_col =['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount','Loan_Amount_Term']
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train[list_col]= sc.fit_transform(X_train[list_col])
    X_validate[list_col]= sc.transform(X_validate[list_col])
    X_test[list_col]= sc.transform(X_test[list_col])
    return X_train, X_validate, X_test

def get_status_heatmap(df):
    '''returns a heatmap with correlations'''
    plt.figure(figsize=(8,12))
    loan_heatmap = sns.heatmap(df.corr()[['Loan_Status']].sort_values(by='Loan_Status',                     ascending=False), vmin=-.5, vmax=.5, annot=True,cmap='flare')
    loan_heatmap.set_title('Features Correlated with Loan Status')


