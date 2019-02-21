# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:35:45 2018

@author: admin123
"""
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#from plotnine import *
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
import gc
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
                GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors                
import os, warnings, pickle, random
import time, datetime, json
from itertools import combinations
from contextlib import contextmanager
gc.enable()

#Path for input data files from Kaggle. All CSVs here will be imported
#PATH = 'C:\\Users\\alonb\\Desktop\\MS Malware 2019\\data\\'
#PATH = 'C:\\Users\\alonb\\Desktop\\MS Malware 2019\\data\\'
PATH = os.path.join(os.getcwd(), 'data')
target_var = 'HasDetections'
id_var = 'MachineIdentifier'

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{}: {:.0f}s".format(title, time.time() - t0))

#Kaggle-compatible log
def log(msg):
    print(msg)
    #Echo for commit log
    if '/kaggle' in os.getcwd():
        os.system('echo %s' % msg)

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def reduce_mem_usage(df, cols = None):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if cols is None: cols = df.columns
        
    for col in cols:
        #print(col)
        col_type = df[col].dtype
        #print(col_type)
        
        if col_type.name in ('object','category'):
            df[col] = df[col].astype('category')
        elif ('datetime' in col_type.name):
            pass
        else:
            #Number
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                #Integer
                if df[col].nunique() < 5:
                    df[col] = df[col].astype('category') 
                elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #Float
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory {0:.1f} --> {1:.1f} MB ({2:.1f}%)'.format(start_mem,end_mem,100 * (end_mem - start_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    #file = 'D:\Documents-Alon\Kaggle\Home Credit\data\HomeCredit_columns_description.csv'
    print('Importing ' + file + ':')
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

#Load datasets
def dir_csv2pickle(path):
    data = {}
    import os
    for file in os.listdir(path):
        fname, ext = os.path.splitext(file)
        if ext == '.csv':
            data[fname] = import_data(path+file)
            with open(path+ '\\' + fname + '.pickle', 'wb') as handle:
                pickle.dump(data[fname], handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data;

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

def outliers_iqr(ys,thresh=1.5):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1    
    lower_bound = quartile_1 - (iqr * thresh)
    lower_bound = max(ys.min(),lower_bound)
    upper_bound = quartile_3 + (iqr * thresh)
    upper_bound = min(ys.max(),upper_bound)
    return lower_bound, upper_bound;
   
def plot_target(df,var,target=[target_var]):
    df = df[[var,target]]
    col_type = df[var].dtype.name
    if (col_type == 'category') or (df[var].nunique()<30):
        #Categorical target
        d2 = df.groupby([target,var]).size() / \
        df.groupby([target]).size()
        d2 = d2.reset_index()
        d2.columns = [target,var,'Freq']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g1 = (ggplot(d2, aes(x=var, y='Freq', fill=target)) + 
                  geom_bar(stat='identity', position='dodge') +
                  theme(axis_text_x = element_text(angle = 90, hjust = 1)))
    else:
        #Numerical target: Detect, remove outliers
        xmin, xmax = outliers_iqr(df[var])
        rows = df[var].shape[0]
        outliers = df[ (df[var] < xmin) | (df[var] > xmax) ].shape[0]
        if outliers > 0:
            print('\n\nRemoved {0:2.1f}% outliers in {1}: {2:,}\n\n'.format( \
                  outliers/rows*100,var,outliers))
        else:
            xmin = df[var].min()
            xmax = df[var].max()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g1 = (ggplot(df, 
                    aes(x=var,group=[target_var],color=[target_var])) + 
                    geom_density() +
                    xlim(xmin, xmax) + 
                    theme_seaborn())
        # Calculate correlation, medians for repaid vs not repaid
        corr = df[target].corr(df[var])
        avg_repaid = df.loc[df[target] == 0, var].median()
        avg_not_repaid = df.loc[df[target] == 1, var].median()
        
        # print correlation + medians
        print('The correlation between %s and the TARGET is %0.4f' % (var, corr))
        print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
        print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
    
    return g1;
 

def load_to_sqlite(data,path=PATH):
    import sqlite3
    import pandas as pd #IO
    #import sqlalchemy
    conn = sqlite3.connect(PATH + 'HCredit.db')
    #cursor = conn.cursor()
    for file in list(data):
        print('Loading '+file+' to sqlite')
        data[file].to_sql(file, conn, index = False,if_exists='replace')
    return;

def dummies(df,exclude=[[id_var],[target_var]]):
    """Convert all categorical variables to dummies / 1-hot encoding"""
    num_cols = list(df._get_numeric_data().columns)
    cat_cols = list(set(df.columns) - set(num_cols) - set(exclude))
    cat_cols = [c for c in cat_cols if df[c].nunique() > 2]    
    df2 = df.drop(columns=cat_cols)
    df3 = pd.get_dummies(df[cat_cols],prefix=cat_cols)
    return pd.concat([df2,df3], axis=1);


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def summary(df):
    """summary stats for a dataframe"""
    #df = data['bureau']
    df3 = df.describe(include='all',percentiles=[.01,.25,.5,.75,.99]).transpose().reset_index()
    df4 = pd.DataFrame(df.dtypes).reset_index()
    df4.columns = ['index','type']
    df5 = pd.merge(left=df4,right=df3,on='index',how='outer')
    missing = missing_data(df).reset_index()
    missing = missing[['index','Percent']]
    df5 = df5.merge(missing,on='index',how='left')
    df5['missing'] = df5['Percent'].map('{:.1f}%'.format)
    del df5['Percent']
    
    if 'freq' in df5.columns:
        freq = df5['freq'] / df.shape[0]
        df5['Top1freq'] = (100*freq).map('{:.1f}%'.format)
    unique = df.agg(['nunique','median']).transpose().reset_index()
    df5 = df5.merge(unique,on='index')
    
    if 'freq' in df5.columns:
        cols = ['index',
                 'type',
                 'missing',
                 'nunique',
                 'top',
                 'Top1freq',
                 'min',
                 '1%',
                 '25%',
                 '50%',
                 'mean',
                 'median',
                 '75%',
                 '99%',
                 'max',
                 'std']
    else:
        cols = ['index',
                 'type',
                 'missing',
                 'nunique',
                 'min',
                 '1%',
                 '25%',
                 '50%',
                 'mean',
                 'median',
                 '75%',
                 '99%',
                 'max',
                 'std']
    df5[cols].to_clipboard(sep=',', index=False)
    return df5[cols];

def prefix(df,prefix,exclude=["SK_ID_CURR","SK_ID_PREV","TARGET"]):
    cols = df.columns
    return np.where(cols.isin(exclude),cols,prefix+'_'+cols);

def round_precision(x):
    if math.isnan(x): return np.nan;
    if x == 0: return 0;
    for digits in range(6):
        y = x % 10 ** digits
        if (y != 0): break
    return digits-1;

def fix_days_365(df):
    """Fix "DAYS" columns with outliers at 365243 --> nan"""
    #df = data['previous_application'].copy()
    shape_start = df.shape[1]
    day_cols = [c for c in df.columns if 'DAYS' in c]
    for col in day_cols:
        if df[col].max() == 365243:
            #print(col)
            newcol = col + '_365'
            df[newcol] = (df[col] == 365243)
            #print((prev[col] == 365243).sum() / prev.shape[0])
            df[col].replace(365243, np.nan, inplace= True)
    newcols = df.shape[1] - shape_start
    if newcols > 0: print('Added {} column(s) for 365 fix'.format(newcols))
    return df;

#directory = 'D:\\Documents-Alon\\Kaggle\\Home Credit\\experiment\\1533067622 LGBMClassifier, AUC=79.59 ensemble'
#result = read_pickle(directory + '\\result.pickle')
#feature_importance_df_ = result['features']
#display_importances(feature_importance_df,directory)

def display_importances(feature_importance_df_, directory):
    cols = feature_importance_df_[["feature", "importance"]
        ].groupby("feature").mean().sort_values(by="importance", ascending=False)[:50].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(directory + '\\lgbm_importances.png')
    plt.close('all')


def display_roc(y, train, oof_preds, folds, directory):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds.split(train,y)):  
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y.iloc[val_idx], oof_preds[val_idx])
        score = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y, oof_preds)
    score = roc_auc_score(y, oof_preds)
    plt.plot(fpr, tpr, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(directory + '\\roc_curve.png')
    
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    precision, recall, thresholds = precision_recall_curve(y, oof_preds)
    score = roc_auc_score(y, oof_preds)
    plt.plot(recall, precision, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Recall / Precision')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(directory + '\\recall_precision_curve.png')
    plt.close('all')
    


def bag_model(df_train, clf, df_test=None, submission=None,
              test_split=True, 
              seed=42, bags=3, verbose=False, 
              foldername='', early_stopping_rounds=50, bootstrap=1, 
              recode=True, random_state=32587,  savedata=True, 
              target_var='HasDetections', id_var='MachineIdentifier'):
    
    from sklearn.model_selection import train_test_split
    with timer('bag_model'):
        model_type = type(clf).__name__
        directory = save_df(df_train, savedata=savedata)
        assert(all(df_train[target_var].notnull()))
        feats = [c for c in df_train.columns if c not in [id_var,target_var]]
        if df_test is not None:
            assert(all(df_test[target_var].notnull()))
            X_train, y_train = df_train[feats], df_train[target_var]
            X_test,  y_test  = df_test[feats], df_test[target_var]
            y_test = y_test.astype('float64') #Otherwise AUC overflows            
            X_test.index = df_test[id_var]
        elif test_split:
            X_train, X_test, y_train, y_test = train_test_split( \
                train[feats], train[target_var], test_size=0.1, random_state=seed)
            X_test.index = df_test[id_var]
        else:
            X_train, y_train = df_train[feats], df_train[target_var]
            X_test,  y_test = pd.DataFrame(), pd.Series()
        print("Train shape: {}, test shape: {}".format(X_train.shape, X_test.shape))
        test_preds = np.zeros((y_test.shape[0],bags))
        
        scores = []
        feature_importance_df = pd.DataFrame()
        if submission is not None:
            sub_preds = np.zeros((submission.shape[0],bags))
        for bag in range(bags):
            if 'random_state' in clf.get_params().keys():
                clf.set_params(random_state = bag * 1000)
            if bootstrap > 1:
                indices = np.random.choice(X_train.index,
                                           X_train.shape[0] * bootstrap)
                print('bootstrapping: X_train shape={}'.format(
                        X_train.loc[indices].shape))
            else:
                indices = X_train.index

            if len(X_test)>0:
                if model_type in ['LGBMClassifier']:
                    clf.fit(X_train.loc[indices], y_train.loc[indices], 
                            eval_set=[(X_train.loc[indices], y_train.loc[indices]), 
                                      (X_test, y_test)], 
                            eval_metric= 'auc', verbose=verbose, 
                            early_stopping_rounds= early_stopping_rounds)
                    test_preds[:,bag] = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1]
                    if submission is not None:
                        sub_preds[:,bag] = clf.predict_proba(submission[feats], num_iteration=clf.best_iteration_)[:, 1]
                else:
                    clf.fit(X_train.loc[indices], y_train.loc[indices])
                    test_preds[:,bag] = clf.predict_proba(X_test)[:, 1]
                    if submission is not None:
                        sub_preds[:,bag] = clf.predict_proba(submission[feats])[:, 1]
                bag_score = roc_auc_score(y_test, test_preds[:,bag])
                bag_score_cum = roc_auc_score(y_test, test_preds.mean(axis=1))
                if 'best_iteration_' in dir(clf):
                    best_iter = clf.best_iteration_
                else:
                    best_iter = None
                scores.append([bag, bag_score, bag_score_cum, best_iter])
                print('Bag %d AUC: %2.4f| Cumulative AUC: %2.4f' % \
                      (bag, bag_score * 100, bag_score_cum * 100) )
            else:
                #No test set - training on full data
                clf.fit(X_train.loc[indices], y_train.loc[indices])
                if submission is not None:
                    sub_preds[:,bag] = clf.predict_proba(submission[feats])[:, 1]
                print('Bag %d of %d' % (bag, bags) )

            filename = directory+ '\\clf%d.pickle' % (bag)
            with open(filename, 'wb') as handle:
                pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            if ('feature_importances_' in dir(clf)):
                bag_importance_df = pd.DataFrame()
                bag_importance_df["feature"] = feats
                bag_importance_df["importance"] = clf.feature_importances_
                bag_importance_df["bag"] = bag + 1
                feature_importance_df = pd.concat([feature_importance_df, bag_importance_df], axis=0)
    
        #Save everything
        with open(directory+ '\\model_params.txt', 'w') as file:
            file.write(str(clf.get_params()))
            #file.write(json.dumps(clf.get_params())) # use `json.loads` to do the reverse
        if len(X_test)>0:
            scores = pd.DataFrame(data=scores,columns=['bag','bag_score','bag_score_cum','best_iteration'])
            scores.to_csv(directory + '\\scores.csv', index=False)

        if ('feature_importances_' in dir(clf)):
            ft = feature_importance_df.pivot(index='feature', columns='bag', values='importance')
            ft['Avg'] = ft.mean(axis=1)
            ft.sort_values(by=['Avg'],ascending=False,inplace=True)
            ft['RankPct']= ft['Avg'].rank(pct=True)
            ft = ft.reset_index()
            ft.to_csv(directory + '\\feature_importance' + str(random.randint(1,1000)) + \
                  '.csv', index=False)
        
        #Save submission file
        if submission is not None:
            bagged_sub_preds = sub_preds.mean(axis=1)
            out_df = pd.DataFrame({id_var : submission[id_var],
                                   target_var : bagged_sub_preds })
            out_df.to_csv(directory + '\\submission.csv', index=False)
        #Save oofs
        if len(X_test)>0:
            oof = pd.DataFrame({id_var: X_test.index, 
                                target_var: test_preds.mean(axis=1) })
            oof.to_csv(directory + '\\oof.csv', index=False)


        if len(X_test)>0:
            os.rename(directory,directory + foldername + ', AUC=%2.2f' % (bag_score_cum*100))
        else:
            os.rename(directory,directory + foldername + ', full train')
        return scores;


    
def kfold_model2(df, clf, num_folds=2, bags=1, CV='kfold', recode=True,
                early_stopping_rounds= 200, verbose=200, foldername=None, 
                resume_experiment = None, fold_only=None,
                random_state=32587, fulldata=False, extraseed=456,
                savedata=True, submission=True, fm_xsample = 0.1,
                target_var='HasDetections',id_vars='MachineIdentifier'):
    import pickle, json
    from time import strftime
    from datetime import datetime
    
    if "fm.Model" in str(type(clf)):
        model_type = "FM"
    else:
        model_type = type(clf).__name__
        
    if recode:
        if (model_type == 'LGBMClassifier') or \
            (str(type(clf)) == "<class 'models.nffm.Model'>") :
            df = df #Do nothing
        elif (model_type in ['CatBoostClassifier','XGBClassifier']):
            df, cat_cols = no_cat(df)            
            df.columns = [c.translate(str.maketrans('][<', '___')) for c in df.columns]
        else: 
            df = no_miss(df) #Includes remove categories
    
    directory = save_df(df,savedata=savedata,resume_experiment=resume_experiment)
        
    # Divide in training/validation and test data
    if df[target_var].dtype.name != 'float64':
        df.loc[:,target_var] = df.loc[:,target_var].astype('float64') #So AUC doesn't overflow
    train_df = df[df[target_var].notnull()]
    test_df = df[df[target_var].isnull()]
    feats = [f for f in train_df.columns if f not in [target_var,id_vars]]
    log("Starting " + model_type + ". Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    
    # Create arrays and dataframes to store results
    scores = []
    best_iter = []
    feature_importance_df = pd.DataFrame()
    oof_preds = np.zeros((train_df.shape[0],bags))
    sub_preds = np.zeros((test_df.shape[0],bags))    

    #Start bag loop        
    for bag in range(bags):
        #if 'random_state' in clf.get_params().keys():
        try:
            clf.set_params(random_state = (random_state * bag + extraseed))
        except:
            log('Could not set random state for model. Using random.seed and np.random.seed')
            random.seed(random_state * bag + extraseed)
            np.random.seed(random_state * bag + extraseed)
        
        # Cross validation model
        if CV=='stratified':
            folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=(random_state))
            split1 = folds.split(train_df[feats], train_df[target_var])
        elif CV=='kfold':
            folds = KFold(n_splits= num_folds, shuffle=True, random_state=(random_state))
            split1 = folds.split(train_df[feats], train_df[target_var])
        elif CV=='adversarial':
            #Adversarial, Scheme A
            folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=(random_state))
            try:
                split1 = folds.split(train_df[feats], is_test_preds['predictions'].values)
            except NameError:
                filepath = os.path.join(PATH, 'df_adversarial.pickle')
                is_test_preds = read_pickle(filepath) #P(obs in train)
                assert(len(is_test_preds)==len(train_df))
                split1 = folds.split(train_df[feats], is_test_preds['predictions'].values)            
        else:
            raise Exception('Unknown CV: %s' % CV)
            break;
            
        #Start fold loop        
        for n_fold, (train_idx, valid_idx) in enumerate(split1):
            #In Kaggle cloud, compute one fold only
            if (fold_only is not None) and (n_fold != fold_only):
                log('Skipping fold %d' % n_fold)
                continue;
            train_x, train_y = train_df[feats].iloc[train_idx], train_df[target_var].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target_var].iloc[valid_idx]
    
            if (model_type == "LGBMClassifier"):
                #print('trace1')                
                #Try loading the model instead of executing it
                model_name = '{}_bag{}_fold{}.pickle'.format(model_type,bag,n_fold+1)
                if os.path.isfile(directory+model_name): 
                    log('Loaded %s. Skipping fit for this fold' % (directory+model_name))
                    clf = read_pickle(directory+model_name)
                else:
                    log('Fitting %s' % (model_name))
                    #Traditional fit disabled for now. No early stopping to accelerate the learning process.
                    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                        eval_metric= 'auc', verbose= verbose, early_stopping_rounds= early_stopping_rounds)
                    #Train 20k rounds, no early stopping
                    #clf.fit(train_x, train_y)
            elif (model_type == "XGBClassifier"):
                #print('trace')
                clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], 
                    eval_metric= 'auc', verbose= 50, early_stopping_rounds= 50)
                clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                    eval_metric= 'auc', verbose= verbose, early_stopping_rounds= early_stopping_rounds)
            elif (model_type == "CatBoostClassifier"):
                bad_nan_cols = [c for c in train_x.columns if \
                     (valid_x[c].isnull().sum()>0) &
                     (train_x[c].isnull().sum() == 0)]
                if len(bad_nan_cols)>0:
                    log(bad_nan_cols)
                    train_x[bad_nan_cols] = train_x[bad_nan_cols].fillna(-999)
                    valid_x[bad_nan_cols] = valid_x[bad_nan_cols].fillna(-999)            
                
                cat_cols_idx = [train_x.columns.get_loc(col) for col in cat_cols]
                #print('catboost fit start')
                clf.fit(train_x, train_y, 
                       eval_set=[(valid_x, valid_y)],
                       #verbose=verbose, metric_period=verbose,
                       #verbose=5, metric_period=5,
                       early_stopping_rounds = 4,
                       use_best_model=True, cat_features=cat_cols_idx)
                #print('catboost fit end')
            #elif (str(type(clf)) == "<class 'models.nffm.Model'>"): #From ctrNet
            elif (model_type=="FM"): #From ctrNet                
                import ctrNet
                #Reset the model at each fold
                hparams = clf.hparams
                clf = ctrNet.build_model(hparams)
                #Sample train_x to accelerate model build
                valid_idx_sample = np.random.choice(valid_idx,size=int(fm_xsample*len(valid_idx)))
                valid_x_sample, valid_y_sample = train_df[feats].iloc[valid_idx_sample], train_df[target_var].iloc[valid_idx_sample]
                
                #train_x = train_x.sample(frac=fm_xsample,random_state=(random_state))
                #train_y = train_y.sample(frac=fm_xsample,random_state=(random_state))
                
                #Then train
                log("Fold %d; train_x: %s; valid_x_sample: %s" % \
                      (n_fold,train_x.shape, valid_x_sample.shape))
                clf.train(train_data=(train_x,train_y), dev_data=(valid_x_sample,valid_y_sample))
                log('train complete')

            else:
                clf.fit(train_x, train_y)

            #Score OOFs and submission
            if (model_type == "LGBMClassifier"):
                #Requires "best_iteration" argument
                oof_preds[valid_idx,bag] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
                if submission:
                    sub_preds[:,bag] += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
            elif (model_type=="FM"): #From ctrNet
                oof_preds[valid_idx,bag] = clf.infer(dev_data=(valid_x,valid_y))
                if submission:
                    sub_preds[:,bag] += clf.infer(dev_data=(test_df[feats], test_df[target_var])) / folds.n_splits
            else: #elif (model_type == "XGBClassifier"):
                oof_preds[valid_idx,bag] = clf.predict_proba(valid_x)[:, 1]
                if submission:
                    sub_preds[:,bag] += clf.predict_proba(test_df[feats])[:, 1] / folds.n_splits
    
            if ('best_iteration_' in dir(clf)):
                best_iter.append(clf.best_iteration_)
            if ('feature_importances_' in dir(clf)):
                if (model_type == "CatBoostClassifier"):
                    fold_importance_df = pd.DataFrame(
                        list(zip(train_x.dtypes.index, 
                            clf.get_feature_importance(
                                Pool(train_x, label=train_y, cat_features=cat_cols_idx)
                            ))),
                        columns=['feature','importance'])
                    missing_feats = [c for c in feats if c not in 
                                     fold_importance_df.feature.values]
                    missing_df = pd.DataFrame({'feature':missing_feats,'importance':0})
                    fold_importance_df = pd.concat([fold_importance_df,missing_df])
                else:
                    fold_importance_df = pd.DataFrame()
                    fold_importance_df["feature"] = feats
                    fold_importance_df["importance"] = clf.feature_importances_
                fold_importance_df["fold"] = n_fold + 1
                fold_importance_df["bag"] = bag
                feature_importance_df = pd.concat(
                        [feature_importance_df, fold_importance_df], axis=0)
            
            model_name = '{}_bag{}_fold{}'.format(model_type,bag,n_fold+1)
            try:
                filepath = os.path.join(directory, model_name + '.pickle')
                with open(filepath, 'wb') as handle:
                    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
                log('Saved pickle to: ' + filepath)
                filepath = os.path.join(directory, model_name + '_params.txt')
                with open(filepath, 'w') as file:
                    file.write(str(clf.get_params()))
            except:
                log('Could not save model')
                #file.write(json.dumps(clf.get_params())) # use `json.loads` to do the reverse
            fold_score = roc_auc_score(valid_y, oof_preds[valid_idx,bag])
            log('%s: Fold %d/%d AUC: %2.4f' % 
                      (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                       n_fold + 1, folds.n_splits, fold_score * 100)
                )
            #End fold loop
        
        bag_score = roc_auc_score(train_df[target_var], oof_preds[:,bag])
        bag_score_cum = roc_auc_score(train_df[target_var], oof_preds.mean(axis=1))
        scores.append([bag, bag_score, bag_score_cum])
        log('%s: Bag %d/%d AUC: %2.4f| Cumulative AUC: %2.4f' % \
                  (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                   bag+1, bags, bag_score * 100, bag_score_cum * 100) 
            )
        #End bag loop
    
    #Save remaining results
    scores = pd.DataFrame(data=scores,columns=['bag','bag_score','bag_score_cum'])
    filepath = os.path.join(directory, 'scores.csv')
    scores.to_csv(filepath, index=False)
    
    #Save submission
    if submission:
        out_df = pd.DataFrame({id_vars : test_df[id_vars], 
                               target_var: sub_preds.mean(axis=1) })
        filepath = os.path.join(directory, 'submission.csv')
        out_df.to_csv(filepath, index=False)
        
        oof = pd.DataFrame({id_vars: train_df.index, 
                       target_var: oof_preds.mean(axis=1) })
        filepath = os.path.join(directory, 'oof.csv')
        oof.to_csv(filepath, index=False)

    if ('feature_importances_' in dir(clf)):
        ft = feature_importance_df.pivot_table( \
                index='feature', columns=['fold','bag'], values='importance')
        ft['Avg'] = ft.mean(axis=1)
        ft.sort_values(by=['Avg'],ascending=False,inplace=True)
        ft['RankPct']= ft['Avg'].rank(pct=True)
        ft = ft.reset_index()
        filepath = 'feature_importance%d.csv' % random.randint(1,1000)
        filepath = os.path.join(directory, filepath)
        ft.to_csv(filepath, index=False)
        #display_importances(feature_importance_df,directory)
    else:
        ft = pd.DataFrame()
  
    new_dirname = '%s_%s_AUC=%2.2f' % (directory, foldername, bag_score_cum*100)
    #Assemble results and return
    result = {'folds' : folds,
              'features' : ft, #feature_importance_df
              'oof_preds' : pd.DataFrame(data=oof_preds,index=train_df.index),
              'sub_preds' : pd.DataFrame(data=sub_preds,index=test_df.index),
              'directory' : new_dirname}
    
    filepath = os.path.join(directory, 'result.pickle')
    with open(filepath, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if (resume_experiment is None) and ('/kaggle/working' not in os.getcwd()):
        os.rename(directory, new_dirname)
    
    #Run the model on the full dataset after folds are done:
    if fulldata:
        best_iter_mean = np.mean(best_iter)
        clf.set_params(n_estimators = int(best_iter_mean)+10)
        log('Running model on full dataset')
        with timer('Running model on full dataset'):
            scores = bag_model(df,clf,verbose=200,bags=bags,seed=random_state,
                               test_split=False)
    return result;

def no_cat(df,exclude_cols = ['MachineIdentifier','HasDetections']):
    #Factorize categories to 0/1/2... for xgb model
    #col_target = [target_var]
    #col_id = [id_var]    
    col_cat = df.select_dtypes(include=['category','bool','object']).columns.values
    col_cat = list(set(col_cat) - set(exclude_cols))
    if len(col_cat) == 0:
        print('No categorical columns found.')
    else:
        print('Factorizing ' + str(len(col_cat)) + ' columns')
        for col in col_cat:
            #print(col + ': ' + str(merged[col].nunique()))
            df[col], uniques = pd.factorize(df[col])
    return df, col_cat;

#df = df2.sample(10000) #.iloc[:,:1000]
def no_miss(df,exclude_cols = ['MachineIdentifier','HasDetections']):
    df, cat_col = no_cat(df, exclude_cols = exclude_cols)
    feats = list(set(list(df)) - set(exclude_cols))
    
    f_nulls = df[feats].isnull().any()
    if any(f_nulls):
        f_nulls = list(f_nulls[f_nulls].index)
        print('Imputing means for missing data in %d features' % len(f_nulls))
        df.loc[:,f_nulls] = df[f_nulls].fillna(df[f_nulls].mean())        
        m = df[f_nulls].mean().isnull()
        if any(m):
            print('Dropping %d features with null means' % m[m].shape[0])
            df = df.drop(columns=(m[m].index.values))
            feats = list(set(list(df)) - set(exclude_cols))
            
    f_inf = np.isinf(df[feats].select_dtypes(['number'])).any()
    #x = list(df[feats].select_dtypes(['number']))
    #[c for c in feats if c not in x]
    if any(f_inf):
        f_inf = list(f_inf[f_inf].index)
        print('Imputing Inf/-Inf values for %d features' % len(f_inf))
        df.loc[:,f_inf] =df[f_inf].astype(np.float32).clip(-1e11,1e11)
    
    assert(all(df[feats].notnull()))
    assert(all(np.isfinite(df[feats])))
    return df

#----------------------------
# Compare datasets
#----------------------------
def col_comp(df1, df2, sample_n=1000,index_col=[id_var]):
    #Find uncorrelated columns from new dataset
    
    #df1 = reduce_mem_usage(df1)
    #df2 = reduce_mem_usage(df2)
    df1 = no_cat(df1)
    df2 = no_cat(df2)
    
    df1.index = df1[index_col]
    df2.index = df2[index_col]
    samp = np.random.choice(df1.index, sample_n, replace=False)
    df1_s = df1.loc[samp]
    df2_s = df2.loc[samp]
    assert all(df1_s.index == df2_s.index)

    #First pass: Correlate with equality
    print('pass 1: exact match')
    matches = []
    for col_df1 in df1.columns:
        for col_df2 in df2.columns:
            if (df1_s[col_df1].dtype.name == df2_s[col_df2].dtype.name):
                if (df1_s[col_df1].dtype == df2_s[col_df2].dtype):
                    if (df1_s[col_df1].equals(df2_s[col_df2])):
                        #print('sample match: ' + col_df1 + '=' + col_df2)
                        if (df1[col_df1].corr(df2[col_df2]) > 0.9999):
                            print('match 1: ' + col_df1 + '=' + col_df2)
                            matches.append((col_df1,col_df2,'corr'))
                            break
        
    matches = pd.DataFrame(data=matches,columns=['df1','df2','match'])
    new_cols = list(set(list(df1.columns)) - set(matches.df1))
    
    #Second pass: Correlate without equality
    print('\n\npass 2: Fuzzy match')
    for col_df1 in df1_s[new_cols].columns:
        #print(col_df1)
        for col_df2 in df2_s.columns:
            if (df1_s[col_df1].corr(df2_s[col_df2]) > 0.9999):
                #print('sample correlated: ' + col_df1 + '=' + col_df2)
                if (df1[col_df1].corr(df2[col_df2]) > 0.9999):
                    print('match 2: ' + col_df1 + '=' + col_df2)
                    matches = matches.append(
                            {'df1': col_df1, 'df2' : col_df2, 'match' : 'corr2'},
                            ignore_index=True
                            )
                    break

    matches['LenDiff'] = abs(matches['df1'].str.len() - matches['df2'].str.len())
    new_cols = list(set(list(df1.columns)) - set(matches.df1))
    
    result =  {'matches' : matches, 
              'new_cols' : new_cols}
    return matches, new_cols



# add noise to y axis to avoid overlapping
def rand_jitter(arr):
    nosie = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr))

def draw_feature_distribution(df, column):
    column_values = df[df[column].notna()][column]
    # group by target
    class_0_values = df[df[column].notna() & (df[[target_var]]==0)][column]
    class_1_values = df[df[column].notna() & (df[[target_var]]==1)][column]
    class_t_values = df[df[column].notna() & (df[[target_var]].isna())][column]        
    print('\n\n', column)
    # for features with unique values >= 10
    if len(df[column].value_counts().keys()) >= 10:
        fig, ax = plt.subplots(1, figsize=(15, 4))
        if df[column].dtype == 'object':
            label_encoder = LabelEncoder()
            label_encoder.fit(column_values)
            class_0_values = label_encoder.transform(class_0_values)
            class_1_values = label_encoder.transform(class_1_values)
            class_t_values = label_encoder.transform(class_t_values)
            column_values = label_encoder.transform(column_values)
            plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, fontsize=12, rotation='vertical')

        ax.scatter(class_0_values, rand_jitter([0]*class_0_values.shape[0]), label='Class0', s=10, marker='o', color='#7ac143', alpha=1)
        ax.scatter(class_1_values, rand_jitter([10]*class_1_values.shape[0]), label='Class1', s=10, marker='o', color='#fd5c63', alpha=1)
        ax.scatter(class_t_values, rand_jitter([20]*class_t_values.shape[0]), label='Test', s=10, marker='o', color='#037ef3', alpha=0.4)
        ax.set_title(column +' group by target', fontsize=16)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title(column +' distribution', fontsize=16)
    else:      
        all_categories = list(df[df[column].notna()][column].value_counts().keys())
        bar_width = 0.25
        
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.set_title(column, fontsize=16)
        plt.xlabel('Categories', fontsize=16)
        plt.ylabel('Counts', fontsize=16)

        value_counts = class_0_values.value_counts()
        x_0 = np.arange(len(all_categories))
        y_0 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_0, y_0, color='#7ac143', width=bar_width, label='class0')

        value_counts = class_1_values.value_counts()
        x_1 = np.arange(len(all_categories))
        y_1 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_1+bar_width, y_1, color='#fd5c63', width=bar_width, label='class1')
        
        value_counts = class_t_values.value_counts()
        x_2 = np.arange(len(all_categories))
        y_2 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_2+2*bar_width, y_2, color='#037ef3', width=bar_width, label='test')
        
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        
        for i, v in enumerate(y_0):
            if y_0[i]+y_1[i] == 0:
                ax.text(i - .08, max(y_0)//1.25,  'Missing in Train', fontsize=14, rotation='vertical')
            else:
                ax.text(i - .08, max(y_0)//1.25,  "{:0.1f}%".format(100*y_0[i]/(y_0[i]+y_1[i])), fontsize=14, rotation='vertical')
        
        for i, v in enumerate(y_1):
            if y_0[i]+y_1[i] == 0:
                ax.text(i - .08, max(y_0)//1.25,  'Missing in Train', fontsize=14, rotation='vertical')
            else:
                ax.text(i + bar_width - .08, max(y_0)//1.25, "{:0.1f}%".format(100*y_1[i]/(y_0[i]+y_1[i])), fontsize=14, rotation='vertical')
 
        for i, v in enumerate(y_2):
            if y_2[i] == 0:
                ax.text(i + 2*bar_width - .08, max(y_0)//1.25, 'Missing in Test', fontsize=14, rotation='vertical')
            else:
                ax.text(i + 2*bar_width - .08, max(y_0)//1.25, str(y_2[i]), fontsize=14, rotation='vertical')
        
        plt.xticks(x_0 + 2*bar_width/3, all_categories, fontsize=16)
        
    plt.show()
   
def find_correlated_cols(df,thresh=0.9999,n_sample=1000):
    cols = [c for c in df.columns if c not in [[id_var],[target_var]]]
    samp = df[cols].sample(n_sample)
    cor = abs(samp.corr())
    cor.loc[:,:] =  np.tril(cor, k=-1)
    cor = cor.stack()
    cor = cor[cor > (thresh*0.98)].reset_index()
    result = []
    for row in range(cor.shape[0]):
        col1 = cor.iloc[row]['level_0']
        col2 = cor.iloc[row]['level_1']
        corr1 = df[col1].corr(df[col2])
        result.append(corr1)
    
    cor['corr_full'] = result
    cor = cor[cor['corr_full'] > thresh]
    cols = list(set.union(set(cor['level_0']),set(cor['level_1'])))
    cor = abs(df[cols].corr())
    cor.loc[:,:] = np.tril(cor, k=-1)
    
    already_in = set()
    clusters = []
    for col in cor:
        perfect_corr = cor[col][cor[col] > thresh].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            clusters.append(perfect_corr)
    
    drop_cols = []
    for grp in range(len(clusters)):    
        drop_cols.append(clusters[grp][1:])
    flatten = lambda l: [item for sublist in l for item in sublist]
    drop_cols = flatten(drop_cols)    
    return {'drop_cols' : drop_cols, 
            'clusters' : clusters, 
            'cor' : cor}

def drop_feats(df,feats, thresh_imp_rank_pct=None, thresh_imp_abs=None,
               thresh_corr=None, verbose=False, tgt_var = 'HasDetections'):
    #Remove columns with low importance according to previous run or with
    # high correlation with other columns in same dataframe
    #Check that all cols in df are covered in importance chart
    #assert set(df) - set(feats['feature']) == {[id_var], [target_var]}  
    start_cols = df.shape[1]

    # Drop if imporance < 10
    drop_cols = pd.DataFrame()
    if thresh_imp_rank_pct is not None:
        drop_cols = feats[feats['RankPct'] <= thresh_imp_rank_pct]['feature']
        thresh_str = "importance rank <= " + str(thresh_imp_rank_pct)
    elif thresh_imp_abs is not None:
        drop_cols = feats[feats['Avg'] <= thresh_imp_abs]['feature']
        thresh_str = "importance <= " + str(thresh_imp_abs)

    if drop_cols.shape[0]>0:
        df.drop(columns=drop_cols,inplace=True,errors='ignore')
        assert set.intersection(set(drop_cols), set(df)) == set() #No unimportant cols
        new_cols = df.shape[1]
        print('Dropped ' + str(start_cols - new_cols) + ' columns with ' + thresh_str)
        if verbose: print(str(drop_cols))
        start_cols = new_cols
    
    # Drop if correlation > 0.99; keep higher importance-feature
    if thresh_corr is not None:
        corr_col = find_correlated_cols(df,thresh=thresh_corr)
        drop_cols = []
        for cluster in corr_col['clusters']:
            #print(cluster)
            feat_view = feats[feats['feature'].isin(cluster)]
            if feat_view.shape[0]>0:
                keep_feat_idx = feat_view['Avg'].argmax()
                keep_feat = feats.iloc[keep_feat_idx]['feature']
            else:
                keep_feat = cluster[0]
            drop_col = [x for x in cluster if x not in keep_feat]
            drop_cols.append(drop_col)
            if verbose: print('Keep: '+keep_feat + '\nDrop:'+ str(drop_col) + '\n')
        flatten = lambda l: [item for sublist in l for item in sublist]
        drop_cols = flatten(drop_cols) 
        df.drop(columns=drop_cols,inplace=True)
        assert set.intersection(set(drop_cols), set(df)) == set() #No corr cols
        new_cols = df.shape[1]
        print('Dropped ' + str(start_cols - new_cols) + \
              ' columns with correlation > ' + str(thresh_corr))
        start_cols = new_cols
    
    #Drop cols with df.nunique==1
    drop_cols = [c for c in df.columns if df[c].nunique() == 1]
    df.drop(columns=drop_cols,inplace=True)
    assert set.intersection(set(drop_cols), set(df)) == set()
    new_cols = df.shape[1]
    print('Dropped ' + str(start_cols - new_cols) + \
              ' columns with df.nunique()==1')
    if verbose: print(str(drop_cols))
    start_cols = new_cols

    #Drop cols with train.nunique==1
    train = df[df[tgt_var].notnull()]
    drop_cols = [c for c in df.columns if train[c].nunique() == 1]
    df.drop(columns=drop_cols,inplace=True)
    assert set.intersection(set(drop_cols), set(df)) == set()
    new_cols = df.shape[1]
    print('Dropped ' + str(start_cols - new_cols) + \
              ' columns with train.nunique()==1')
    if verbose: print(str(drop_cols))
    start_cols = new_cols
    
    return df;

def read_pickle(path):
    with open(path, "rb") as input_file:
        df = pickle.load(input_file)
    return df;

#Save to disk
def save_df_data(df,name='', cols = None, incl_train_test = True, 
            incl_df = True, reduce_mem = False):
    if cols is None: 
        cols = list(df)
    else:
        cols = list(set(cols + ['MachineIdentifier','HasDetections']))
    if reduce_mem:
        df = reduce_mem_usage(df) #All columns, not just selected ones
    if incl_df:
        df_path = PATH+ '\\df_'+name+'.pickle'
        print('saving df to: ' + df_path)
        with open(df_path, 'wb') as handle:
            pickle.dump(df[cols], handle, protocol=pickle.HIGHEST_PROTOCOL)
    if incl_train_test:
        train = df.loc[df.HasDetections.notnull(),cols]
        test = df.loc[df.HasDetections.isnull(),cols]
        
        train_path = PATH+ '\\train_'+name+'.pickle'
        print('saving train to: ' + train_path)
        with open(train_path, 'wb') as handle:
            pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        test_path = PATH+ '\\test_'+name+'.pickle'
        print('saving test to: ' + test_path)
        with open(test_path, 'wb') as handle:
            pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        del train, test

#Now cross-platform for Kaggle kernels
#On local PC: PATH is D:/.../Kaggle/<Tournament>/data
#On kaggle, PATH is /kaggle/working
# We go up one folder and over to ../experiment
def save_df(df, experiment = '', savedata=True, resume_experiment = None):
    #print('in save_df - trace3')
    if ('/kaggle' in os.getcwd()):
        print('In Kaggle cloud')
        directory = '/kaggle/working';
    elif resume_experiment is not None:
        #Resume experiment with existing saved models in pickle files
        directory = os.path.join(PATH, os.pardir, 'experiment', resume_experiment)
        assert(os.path.exists(directory))
        directory = os.path.abspath(directory)
        return directory;
    else:
        #Local
        directory = os.path.join(PATH, os.pardir, 'experiment',
                                 '%d_%dc_%s' % (int(time.time()), df.shape[1], experiment))
        directory = os.path.abspath(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
    path2 = os.path.join(directory, 'data.pickle')
    if savedata:
        with open(path2, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saving to : ' + directory)
    return directory;

def load_last_df(experiment='1531843816 LGBMClassifier, AUC=79.47'):
    path = PATH + '..\\experiment\\' + experiment

    with open(path + '\\data.pickle', "rb") as input_file:
        df = pickle.load(input_file)
    #del df['index']
    #df.index = df.SK_ID_CURR
    return df;

def get_ensemble(folder, runs, id_var = 'MachineIdentifier', 
                 target_var = 'HasDetections', df_pickle = 'df_orig.pickle'):
    try:
        oof = read_pickle(PATH + 'train_IDs.pickle')
        sub = read_pickle(PATH + 'test_IDs.pickle')
    except:
        #First time run only: Save IDs, targets to pickle
        df = read_pickle(PATH + df_pickle)
        save_df_data(df[[id_var, target_var]], incl_train_test = True, 
                     incl_df=False, name='IDs')
        oof = read_pickle(PATH + 'train_IDs.pickle')
        sub = read_pickle(PATH + 'test_IDs.pickle')

    #run = runs[1]
    for run in runs:
        with open(folder + '\\' + run + '\\result.pickle', "rb") as input_file:
            result = pickle.load(input_file)
            
        oof_preds = result['oof_preds']
        #if oof_preds.shape[0] != expected_nrows: continue
        if isinstance(oof_preds,pd.Series):
            oof[run] = oof_preds.values
        elif isinstance(oof_preds,pd.DataFrame):
            oof[run] = oof_preds.mean(axis=1).values #Average multiple runs
            #TODO: Add row-level stats
        elif isinstance(oof_preds,np.ndarray):
            oof[run] = oof_preds
        
        sub_preds = result['sub_preds']
        if isinstance(sub_preds,pd.Series):
            sub[run] = sub_preds.values
        elif isinstance(sub_preds,pd.DataFrame):
            sub[run] = sub_preds.mean(axis=1).values #Average multiple runs
        elif isinstance(sub_preds,np.ndarray):
            sub[run] = sub_preds
    
    assert(all(oof.columns == sub.columns))
    out = pd.concat([oof,sub],axis=0,sort=False)
    return out;



def mean_enc(df,cat_col,tgt_col='HasDetections',method='kfold'):
    assert(cat_col in list(df))
    assert(tgt_col in list(df))
    if df[tgt_col].dtype.name != 'float32':
        df[tgt_col] = df[tgt_col].astype(np.float32)
    globalmean = df[tgt_col].mean()
    mean_enc_col = df[tgt_col].copy()
    mean_enc_col[:] = globalmean
    if method=='exp':
        cumsum = df.groupby(cat_col)[tgt_col].cumsum() - df[tgt_col]
        cumcnt = df.groupby(cat_col)[tgt_col].cumcount()
        mean_enc_col = cumsum / cumcnt
    elif method=='kfold':
        from sklearn.model_selection import KFold    
        folds = KFold(n_splits=5, shuffle=True,random_state=123)
        for tr_ind, val_ind in folds.split(df):
            X_tr, X_val = df.iloc[tr_ind], df.iloc[val_ind]
            X_tr_means = X_tr.groupby(cat_col)[tgt_col].mean()
            X_val_means = X_val[cat_col].map(X_tr_means)
            mean_enc_col.iloc[val_ind] = X_val_means
    else:
        print('unknown method' + method)
    mean_enc_col = mean_enc_col.fillna(globalmean)
    return mean_enc_col.astype('float16');

def get_best_iter(folder):
    best_iter = []
    for f in os.listdir(directory):
        if '.pickle' in f and 'bag' and 'fold' in f:
            clf = read_pickle(directory + '\\' + f)
            best_iter.append(clf.best_iteration_)
    best_iter_mean = np.mean(best_iter)
    return best_iter_mean

#Write results to disk
def save_model(data,oof_preds,sub_preds,auc=0, name='', folder=None,
               savedata=False):
    if folder is None:
        folder = save_df(data,savedata=savedata)
    if len(oof_preds.shape) == 2:
        oof_preds = oof_preds.mean(axis=1)
    result = {    'oof_preds' : oof_preds,
                  'sub_preds' : sub_preds}
    train_idx = data[data[target_var].notnull()].index
    with open(folder + '\\result.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    test = data[data[target_var].isnull()]
    out_df = pd.DataFrame({id_var: test[id_var], 
                           target_var: sub_preds })
    out_df.to_csv(folder + '\\submission.csv', index=False)
    
    oof = pd.DataFrame({id_var: train_idx, 
                   target_var: oof_preds })
    oof.to_csv(folder + '\\oof.csv', index=False)    
    
    new_dirname = '%s %s, AUC=%2.2f' % (folder, name, auc*100)
    os.rename(folder, new_dirname)
    return new_dirname

def make_interactions(df,interactions, rank=False):
    if rank:
        cols = list(set(list(interactions.f1) + list(interactions.f2)))
        df[cols] = df[cols].rank(pct=True)    
    #Input DF format: f1, f2, op, corr_tgt
    df2 = np.zeros(shape=(len(df),len(interactions)))
    for f in range(len(interactions)):
        if f % 250 == 0:
            print(f)
        f1, f2, op = interactions.iloc[f,:3]
        new_f = new_feature(df,f1,f2,op)
        df2[:,f] = new_f
    df2 = pd.DataFrame(df2, index=df.index, columns=interactions['col'])
    return df2

def new_feature(df,f1,f2,op='minus'):
    if op=='minus':
        new_f = df[f1] - df[f2]
    elif op=='plus':
        new_f = df[f1] + df[f2]
    elif op=='mult':
        new_f = df[f1] * df[f2]
    elif op=='div':
        new_f = df[f1] / (df[f2] + 1e-6)
    elif op=='(a-b)*a':
        new_f = (df[f1] - df[f2]) * df[f1]
    return new_f

def find_interaction(df, eps=0.01,shortnames=False,rank=False,
                     operators = ['minus','plus','mult','div','(a-b)*a']):
    #Identify 2way interactions in df
    # return: 1) summary table 2) interaction feature themselves
    
    feats = [c for c in list(df._get_numeric_data()) \
                if c not in [id_var,target_var]]
    combos = len(feats) * (len(feats)-1)
    log('%d numeric features; %d 2-way interactions x %d operators = %d to search' % 
        (len(feats), combos, len(operators), combos * len(operators)))
    
    if rank:
        df[feats] = df[feats].rank(pct=True)
   
    #df_s = df.sample(n=400000, random_state=42)
    train = df[df[target_var].notnull()].sample(n=100000, random_state=42)
    train_y = train[target_var]
    
    interactions = [] #Interactions summary
    new_feats = [] #new feature table
    for op in operators:
        print('Searching for %s interactions' % op)
        pairs = enumerate(combinations(list(feats),2))
        #break
        for num, (f1, f2) in pairs:
            #break
            if num % 100 == 0: #250 == 0:
                print('Feature %d' % (num))
            if  ('%s_%s_%s'%(f1,op,f2)) in list(feats) or \
                ('%s_%s_%s'%(f2,op,f1)) in list(feats):
                print('Exists: %s_%s_%s'%(f1,op,f2))
                continue
            new_f = pd.DataFrame(new_feature(train,f1,f2,op))
            corr_tgt = new_f.corrwith(train_y) #TODO: Switch to spearman rank
            if math.isnan(corr_tgt[0]) or (abs(corr_tgt[0]) < eps):
                continue
            corr_self = pd.concat([train[[f1,f2]],new_f],axis=1).corr()
            if max(abs(corr_self.loc[0].iloc[0:2]))>0.90:
                continue
            interactions.append([f1, f2, op, abs(corr_tgt[0])])
            #new_f_full = pd.DataFrame(new_feature(df,f1,f2,op))
            new_feats.append(new_f)
    interactions = pd.DataFrame(interactions,columns=['f1','f2','op','corr_tgt'])
    interactions = interactions.sort_values(by='corr_tgt', ascending=False)
    #print('trace')
    new_feats = pd.concat(new_feats,axis=1)
    if shortnames:
        cols = list(interactions.f1.str[:10]+'_'+interactions.op +'_'+interactions.f2.str[:10])
    else:
        cols = list(interactions.f1+'_'+interactions.op+'_'+interactions.f2)
    interactions['col'] = cols
    new_feats.columns = cols
    
    #Find least correlated
    return interactions, new_feats

def knn_feats_kfold(train_x, train_y, test_x, k_list=[3,10], num_folds=2,
                  metrics = ['minkowski','braycurtis','manhattan'],
                  random_state = 32587):
    #k_list = [3,10,100,1000,2000,5000]    
    knn_df = []
    for metric in metrics:
        metric_df = []
        NN = NearestNeighbors(n_neighbors=max(k_list)+1, 
                              metric=metric, n_jobs=-1)        
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=random_state)
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_x, train_y)):
            fold_trx, fold_try = train_x.iloc[train_idx], train_y.iloc[train_idx]
            fold_validx, fold_validy = train_x.iloc[valid_idx], train_y.iloc[valid_idx]
            NN.fit(fold_trx)
            with timer('Metric %s Fold %d' % (metric,n_fold+1)):
                fold_df = make_knn_feats(NN, fold_validx, fold_try.values, k_list)
            metric_df.append(fold_df)
            #End fold loop
            
        metric_df = pd.concat(metric_df,axis=0) #Join folds one after the other
        assert(set(metric_df.index) == set(train_x.index))
        NN.fit(train_x)
        with timer('Metric %s submission' % (metric)):
            sub_df = make_knn_feats(NN, test_x, train_y.values, k_list)
        assert(set(sub_df.index) == set(test_x.index))
        metric_df = pd.concat([metric_df,sub_df],axis=0)
        metric_df.columns = [metric + '_' + c for c in metric_df.columns]
        knn_df.append(metric_df)
        #End metric loop
    knn_df = pd.concat(knn_df,axis=1) #Join metrics side by side
    return knn_df

def make_knn_feats(NN, df, y_train, k_list, eps=1e-6, 
                   big_dist = None):
    
    if big_dist is None: big_dist = max(k_list)*100
        
    NN_output = NN.kneighbors(df) #Model must be fit in advance
    
    # rows = rows in X, start w/0
    # columns = indices of neighbors, start w/0
    neighs = NN_output[1] #[0]
    
    # rows = rows in X, start w/0
    # columns = distances to neighbors
    neighs_dist = NN_output[0] #[0] 
    
    # rows = rows in X, start w/0
    # columns = ground-truth labels of neighbors
    neighs_y = y_train[neighs].astype(np.int8)

    #Don't consider the point itself in the train set (leak)
        #Implementation: The mask will drop one neighbor from each row
    idx = np.array(range(neighs.shape[0]))[:,None]   
    self_neighbor = (neighs == idx)
    no_self_neighbor_rows = np.logical_not(self_neighbor.any(axis=1))
    if any(no_self_neighbor_rows): #If no self-neighbor in row, drop last neighbor
        print('Dropping last neighbor for %d rows' % sum(no_self_neighbor_rows))
        self_neighbor[no_self_neighbor_rows,self_neighbor.shape[1]-1] = True
    assert(np.all(self_neighbor.sum(axis=1))==1) #Drop exactly one in each row
    x_self_neighbor = np.logical_not(self_neighbor) #Include all but self neighbor
    
    drop1col = (neighs.shape[0],neighs.shape[1]-1)
    neighs = neighs[x_self_neighbor].reshape(drop1col)
    neighs_dist = neighs_dist[x_self_neighbor].reshape(drop1col)
    neighs_y = neighs_y[x_self_neighbor].reshape(drop1col)
    assert(neighs.shape[1] == max(k_list))
    assert(neighs_dist.shape[1] == max(k_list))
    assert(neighs_y.shape[1] == max(k_list))

    #Confirm there are no self-neighbors (no points match the index id)
    idx = np.array(range(neighs.shape[0]))[:,None]
    self_neighbor = (neighs == idx)
    assert(np.all(self_neighbor) == False)
    
    return_list = []
    feat_names = []
    #eps = 1e-6
    #big_dist = max(k_list)*100
    
    #1. Fraction of objects of every class.
    #   It is basically a KNNСlassifiers predictions
    for k in k_list:                                         
        feat = np.apply_along_axis(func1d = np.bincount, axis=1, 
                                arr=neighs_y[:,:k], minlength=2) #2 classes
        feat = feat / feat.sum(axis=1, keepdims=True)
        assert(feat.shape == (neighs.shape[0],2))
        return_list += [feat]
        feat_names += ['Target0Pct_k%d' % k, 'Target1Pct_k%d' % k]


    #2. Same label streak: the largest number N, 
    #       such that N nearest neighbors have the same label
    #   Alon: Simplified to: Position of nearest neighbor with target=1, else 999
    #   TODO: Extend to multi-class
    feat = np.where(neighs_y.max(axis=1) == 1,
                    np.argmax(neighs_y==1, axis=1),
                    big_dist)
    feat = feat.reshape(-1,1) #Convert to 2d array so it can be stacked
    assert(feat.shape == (neighs.shape[0],1))
    return_list += [feat]
    feat_names += ['Nearest_Target1_Pos']

    
        
    #3. Minimum distance to objects of each class
    #   Find the first instance of a class and take its distance as features.
    feat1 = np.argmax(neighs_y==1, axis=1) #First neighbor with target=1
    feat2 = neighs_dist[np.arange(len(neighs_dist)), feat1] #distance to that neighbor    
    feat = np.where(neighs_y.max(axis=1) == 1, #If target=1 exists in row...
                     feat2, #Show distance to that neighbor
                     big_dist) #Otherwise dummy value
    feat = feat.reshape(-1,1) #Convert to 2d array so it can be stacked
    assert(feat.shape == (neighs.shape[0],1))
    return_list += [feat]
    feat_names += ['Nearest_Target1_Dist']
    

    #4. Minimum *normalized* distance to objects of each class
    #       As 3. but we normalize (divide) the distances
    #       by the distance to the closest neighbor.
    #       Do not forget to add self.eps to denominator
    feat2 = feat2 / (neighs_dist[:,0] + eps) # divide by dist to first neighbor
    feat = np.where(neighs_y.max(axis=1) == 1, #If target=1 exists in row...
                     feat2, #Show distance to that neighbor
                     big_dist) #Otherwise dummy value
    feat = feat.reshape(-1,1) #Convert to 2d array so it can be stacked
    assert(feat.shape == (neighs.shape[0],1))
    return_list += [feat]
    feat_names += ['Nearest_Target1_Dist_Norm']


    #    5. 
    #      5.1 Distance to Kth neighbor
    #          Think of this as of quantiles of a distribution
    #      5.2 Distance to Kth neighbor normalized by 
    #          distance to the first neighbor
    for k in k_list:            
        feat_51 = neighs_dist[:,k-1]
        feat_52 = neighs_dist[:,k-1] / (np.add(neighs_dist[:,0], eps))
        feat_51 = feat_51.reshape(-1,1) #Convert to 2d array so it can be stacked
        feat_52 = feat_52.reshape(-1,1) #Convert to 2d array so it can be stacked
        return_list += [feat_51, feat_52]
        feat_names += ['Dist_to_k%d' % k, 'Dist_to_k%d_Norm' % k]
    
    
    #    6. Mean distance to neighbors of each class for each K from `k_list` 
    #           For each class select the neighbors of that class among K nearest neighbors 
    #           and compute the average distance to those objects
    #           
    #           If there are no objects of a certain class among K neighbors, set mean distance to 999
    #           
    #       You can use `np.bincount` with appropriate weights
    #       Don't forget, that if you divide by something, 
    #       You need to add `self.eps` to denominator
    for c in [0,1]:        
        feat1 = np.where(neighs_y==c,neighs_dist,np.nan)
        for k in k_list:            
            feat = np.nanmean(feat1[:,:k],axis=1)
            feat[np.isnan(feat)] = big_dist
            feat = feat.reshape(-1,1) #Convert to 2d array so it can be stacked
            assert(feat.shape == (neighs.shape[0],1))
            return_list += [feat]
            feat_names += ['MeanDist_Target%d_k%d' % (c,k)]
    
    #      7. Distance to nearest object
    feat = neighs_dist[:,0].reshape(-1,1)
    assert(feat.shape == (neighs.shape[0],1))
    return_list += [feat]
    feat_names += ['NearestNeighbor_Dist']
    
    knn_feats = np.hstack(return_list)
    knn_feats = pd.DataFrame(knn_feats,index=df.index)
    #knn_feats.columns = ['knn'+str(c) for c in knn_feats.columns]
    #print('Feat_names' + str(feat_names))
    #print(knn_feats.shape)
    knn_feats.columns = feat_names
    return knn_feats

def colwise_auc(train,verbose=False):
    result = []
    runs = [f for f in train.columns if f not in [[target_var],[id_var]]]
    for run in runs:
        score = roc_auc_score(train[target_var],train[run])*100
        if verbose: print('RUN: ' + run + '\t\t AUC: %2.3f' % score)
        result.append([run,score])
    result = pd.DataFrame(result, columns=['run','score'])
    result = result.sort_values(by='score',ascending=False)
    result.index = result['run']
    return result

#Get a 10% sample with a few train records
def df_sample(df, frac=0.1):
    train = df[df.HasDetections.notnull()].sample(frac=frac)
    test = df[df.HasDetections.isnull()].sample(frac=0.0001)
    df_s = pd.concat([train,test],sort=False)    
    del train, test
    return df_s;


#Drop columns with high cardinality <-- Winner
def drop_hi_card(df, max_levels=10):
    cols = [c for c in list(df) if '_FreqEnc' not in c and '_TgtEnc' not in c and
            c not in ['MachineIdentifier','HasDetections']]
    uniq =  df[cols].nunique(dropna=False)
    hi_card_cols = [c for c in uniq[uniq>max_levels].index if 
                    df[c].dtype.name=='category' or 'Identifier' in c]
    lo_card = df[[c for c in df.columns if c not in hi_card_cols]]
    return lo_card, hi_card_cols;

#Group rare categories together
    #Input: Categorical column (could be numeric)
    #Output: All rare categories replaced with replace_value
def rare_cat_group(ser, min_freq=0.0001):
    #ser = df['CityIdentifier']
    #min_count = -1
    if ('int' in ser.dtype.name) or ('float' in ser.dtype.name):
        replace_value = ser.min() - 1e5
    else:
        replace_value = '_Other'
    counts = ser.value_counts()
    min_count = len(ser) * min_freq
    #np.where(counts < 1533806,1,0)
    to_remove = list(counts[counts <= min_count].index)
    #to_remove = counts[counts <= min_count].index.values
    pd.options.mode.chained_assignment = None
    ser[ser.isin(to_remove)] = replace_value
    pd.options.mode.chained_assignment = 'warn'
    return ser;

#Drop one feature at a time, check if score improves.
#Returns scores at each iteration + features which - when dropped - improve the score
def RFE(train, clf, feats = None, verbose=1, 
        early_stopping_rounds=50, tgt='HasDetections'):
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(train,test_size=0.25, random_state=42)
    from sklearn.metrics import roc_auc_score
    scores = []
    if feats is None: 
        feats = [c for c in list(df_train) if c not in ['MachineIdentifier',tgt]]
    #assert(all(df_train.columns = df_test.columns)
    best_feats = feats
    #Baseline to determine # of iterations (estimators)
    df_test.loc[:,tgt] = df_test.loc[:,tgt].astype('float64') #Otherwise roc_auc fails
    
    clf.fit(df_train[feats], df_train[tgt], 
            eval_set=[(df_train[feats], df_train[tgt]), (df_test[feats], df_test[tgt])], 
            eval_metric= 'auc', verbose=-1, 
            early_stopping_rounds=early_stopping_rounds)
    preds = clf.predict_proba(df_test[feats], num_iteration=clf.best_iteration_)[:, 1]
    best_score = roc_auc_score(df_test[tgt], preds)
    best_iter = clf.best_iteration_
    clf.set_params(n_estimators = best_iter) #best_iter
    scores.append(['Baseline: All Features',best_score,feats])
    print('Base score: %2.4f after %d iterations' % (100*best_score, best_iter))
    
    #Drop one feature at a time, check if score improves
    for idx, f_drop in enumerate(feats):
        if (idx % verbose == 0):
            print('%d / %d: %s' % (idx, len(feats), f_drop))
        best_feats = [c for c in best_feats if c != f_drop]
        clf.fit(df_train[best_feats], df_train[tgt], verbose=-1)
        preds = clf.predict_proba(df_test[best_feats], num_iteration=clf.best_iteration_)[:, 1]
        score = roc_auc_score(df_test[tgt], preds)
        scores.append([f_drop,score,best_feats])
        if score >= best_score:
            print('Dropping %s improved score to: %2.4f' % (f_drop, 100*score))
            best_score = score
        elif score >= (best_score - 1e-5):
            print('Dropping %s only hurt score by: %2.6f' % (f_drop, 100*(score-best_score)))
        else:
            best_feats.append(f_drop)
        #imp = pd.DataFrame({'feats' : feats_selected, 'importance' : clf.feature_importances_})
    dropped_feats = [c for c in feats if c not in best_feats]
    scores = pd.DataFrame(scores)
    sel_cols = [c for c in list(df_train) if c not in dropped_feats]
    
    #Save everything
    directory = save_df(df_train,savedata=False,experiment='RFE')
    scores.to_csv(os.path.join(directory, 'scores.csv'))
    with open(os.path.join(directory, 'dropped_feats.txt'), 'w') as file:
        file.write(str(dropped_feats))
    with open(os.path.join(directory, 'selected_cols.json'), 'w') as file:
        json.dump(sel_cols, file)    
    return scores, dropped_feats;
