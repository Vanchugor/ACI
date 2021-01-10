import numpy as np
import pandas as pd
from pandas.plotting import andrews_curves
import seaborn as sns
import re
import time
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def init_check(df):
    columns = df.columns    
    lst = []
    for feature in columns : 
        dtype = df[feature].dtypes
        num_null = df[feature].isnull().sum()
        num_unique = df[feature].nunique()
        lst.append([feature, dtype, num_null, num_unique])
    
    check_df = pd.DataFrame(lst)
    check_df.columns = ['feature','dtype','num_null','num_unique']
    return check_df

def lower_replace(x): 
    if type(x) == np.str:
        for i in '!@#$%^&*()_+-:;.,/':
            x = x.replace(i, '')
        x = x.lower() 
        return x
    else: return x
    

def get_data(path):
    labels_col = ['age', 'workclass', 'final_weight', 'education', 'education_num', 'marital_status', 
              'occupation', 'relationship', 'race', 'sex', 'cap_gain', 'cap_loss', 
              'hours_per_week', 'native_country', 'target']

    data = pd.read_csv(path, sep=',', skipinitialspace=True, 
                       names=labels_col, comment='.', error_bad_lines=False)



    target_values = data['target'].value_counts()

    data = data.applymap(lambda x: lower_replace(x))

    bins = [0, 22, 49, 65, 100]
    age_names = ['young', 'middleaged', 'senior', 'old']
    data['age_new'] = pd.cut(data['age'], bins, labels=age_names)


    data['workclass_new'] = data['workclass']
    data['workclass_new'].replace(['stategov', 'federalgov', 'localgov'], 'govemp', inplace=True)
    data['workclass_new'].replace(['selfempnotinc', 'selfempinc'], 'selfemp', inplace=True)
    data['workclass_new'].replace(['withoutpay', 'neverworked'], 'nopay', inplace=True)
    data['workclass_new'].replace('?', 'unknown', inplace=True)


    data['education_new'] = data['education']
    data['education_new'].replace(['5th6th','10th','1st4th','preschool','12th','9th','7th8th','11th'], 'school', inplace=True)
    data['education_new'].replace(['assocacdm','assocvoc'], 'assoc', inplace=True)
    data.drop(['education'], axis=1, inplace=True)


    data['native_country_new'] = data['native_country']
    data['native_country_new'].replace(['unitedstates', 'south', 'outlyingusguamusvietc'], 'us', inplace=True)
    data['native_country_new'].where(data['native_country_new'] == 'us', 'nonus', inplace=True)


    data['marital_status_new'] = data['marital_status']
    data['marital_status_new'].replace(['marriedafspouse', 'marriedcivspouse'], 'married', inplace=True)
    data['marital_status_new'].where(data['marital_status_new'] == 'married', 'single', inplace=True)


    data['marital_status'].replace(['sales'], np.nan, inplace=True)
    data['relationship'].replace(['sales', 'white'], np.nan, inplace=True)


    data['relationship_new'] = data['relationship']
    data['relationship_new'].replace(['husband', 'wife'], 'spouse', inplace=True)


    data['occupation'].replace(['husband'], np.nan, inplace=True)


    data['occupation_new'] = data['occupation']
    data['occupation_new'].replace(['?'], 'unknown', inplace=True)
    data['occupation_new'].replace(['admclerical','execmanagerial','techsupport'], 'whitecollar', inplace=True)
    data['occupation_new'].replace(['protectiveserv', 'armedforces'], 'protect', inplace=True)
    data['occupation_new'].replace(['otherservice', 'privhouseserv'], 'service', inplace=True)
    data['occupation_new'].replace(['handlerscleaners', 'craftrepair', 'transportmoving',
                                    'farmingfishing', 'machineopinspct'], 'bluecollar', inplace=True)


    data['race_new'] = data['race']
    data['race_new'].replace(['asianpacislander','amerindianeskimo'], 'other', inplace=True)


    data['hours_per_week_new'] = data['hours_per_week']
    data['hours_per_week_new'].where([str.isnumeric(str(x).replace('.0', '')) for x in data['hours_per_week_new']], 
                                     np.nan, inplace=True)
    data['hours_per_week_new'] = data['hours_per_week_new'].astype(np.float)
    data['hours_per_week_new'].where(data['hours_per_week_new'] <= 100, np.nan, inplace=True)


    bins = [0, 38, 41, 61, 100]
    hpw_names = ['undernorm', 'norm', 'overnorm', 'overwork']
    data['hours_per_week_new'] = pd.cut(data['hours_per_week_new'], bins, labels=hpw_names)


    data['cap_gain'].where(data['cap_gain'] >= 0, np.nan, inplace=True)
    data['cap_loss'].where(data['cap_loss'] >= 0, np.nan, inplace=True)
    data = data.assign(cap_gain_loss=data['cap_gain']-data['cap_loss'])



    maximum = data['final_weight'].max()
    minimum = data['final_weight'].min()

    abs_max = 10000000
    data['final_weight'] = data['final_weight'].where(data['final_weight'] < abs_max, np.nan)
    data['final_weight'] = data['final_weight'].where(data['final_weight'] >= 0, np.nan)


    maximum = data['final_weight'].max()
    minimum = data['final_weight'].min()
    span = maximum - minimum
    N = 15
    labels = list(range(1, N+1))
    data['eqwidth_final_weight'], bins = pd.cut(data['final_weight'], N, labels=labels, retbins=True)
    bins = [[i+1, int(bins[i])] for i in range(len(bins))]


    val_c = data['final_weight'].value_counts()
    idx = np.expand_dims(val_c.values.tolist(), axis=1)
    vals = np.expand_dims(val_c.index.tolist(), axis=1)
    val = pd.DataFrame(np.concatenate((vals, idx), axis=1), columns=['vals', 'idx'])
    val['idx'] -= 1

    dict_val = dict(val.values)

    data['final_weight_new'] = data['final_weight']
    data['final_weight_new'] = data['final_weight_new'].map(dict_val, na_action='ignore')


    fw = data[['final_weight','final_weight_new']]
    fw_sort1 = fw.sort_values(by=['final_weight_new'])
    fw_sort1.columns = ['fw1', 'rep1']
    fw_sort2 = fw.sort_values(by=['final_weight'])
    fw_sort2.columns = ['fw2', 'rep2']

    max_rep = len(data['final_weight_new'].value_counts()) - 1
    maxima = []
    for i in range(max_rep+1):
        connectedness_i = data['final_weight'][data['final_weight_new'] == i]
        maxima.append(connectedness_i.max())

    maxima = dict(enumerate(maxima))


    bins = [0, 371888, 933222, abs_max]
    cntd_names = ['high', 'low', 'no']
    data['connectedness_variation'] = pd.cut(data['final_weight'], bins, labels=cntd_names)

    data['final_weight_new'] = data['connectedness_variation']
    data.drop(['connectedness_variation', 'eqwidth_final_weight'], axis=1, inplace=True)
    
    to_delete = data['target'][data['target'].isna()].index.to_list()
    data = data.drop(to_delete, axis=0)
    
    return data

def impute_encode_scale(data, imp, ore, ohe, minmax):  
    labels_col = ['age', 'workclass', 'final_weight', 'education_num', 'marital_status', 
                  'occupation', 'relationship', 'race', 'sex', 'cap_gain', 'cap_loss', 
                  'hours_per_week', 'native_country', 'target']
    labels_col_new = ['age_new', 'workclass_new', 'education_new', 'native_country_new', 'marital_status_new', 
                      'relationship_new', 'occupation_new', 'race_new', 'sex', 'hours_per_week_new', 
                      'cap_gain_loss', 'final_weight_new', 'target']

    big_X = data[labels_col]
    X = data[labels_col_new]

    big_y = big_X.pop('target')
    y = X.pop('target')
    
    X.loc[:, 'age_new'] = X['age_new'].map({'middleaged': 1, 'senior': 2, 'young': 0, 'old': 3}, na_action='ignore')
    X.loc[:, 'education_new'] = X['education_new'].map({'bachelors':4, 'hsgrad':1, 'school':0, 
                                                         'masters':5, 'somecollege':2, 'assoc':3, 
                                                         'doctorate':7, 'profschool':6}, na_action='ignore')
    X.loc[:, 'hours_per_week_new'] = X['hours_per_week_new'].map({'norm':1, 'undernorm':0, 
                                                                   'overnorm':2, 'overwork':3}, na_action='ignore')
    X.loc[:, 'final_weight_new'] = X['final_weight_new'].map({'high':0, 'low':1, 'no':2}, na_action='ignore')

    X = pd.DataFrame(imp.transform(X), columns=labels_col_new[:-1])
    
    encode_col = ['workclass_new','native_country_new','marital_status_new','relationship_new', 
                  'occupation_new','race_new','sex']
    
    encode = X.loc[:, encode_col]

    encode = pd.DataFrame(ore.transform(encode), columns=encode_col)
    X.loc[:, encode_col] = encode.loc[:, encode_col]
    X = X.astype(np.float)

    y = y.map({'>n':1, '<=n':0})
    
    save_for_rfc = X.copy()
    
    #X.drop(['final_weight_new'], axis=1, inplace=True)
    
    encode = X.loc[:, encode_col]

    encode_onehot_col = ohe.get_feature_names(encode_col)
    encode = pd.DataFrame(ohe.transform(encode), columns=encode_onehot_col)

    encode.columns = [x[:-2] for x in encode.columns]
    
    X.drop(encode_col, axis=1, inplace=True)
    
    cols = X.columns
    X = pd.DataFrame(minmax.transform(X), columns=cols)
    
    X = pd.concat([X, encode], axis=1)
    
    return X, y, big_X, big_y, save_for_rfc
    
    
    
    
    
    