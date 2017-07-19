import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesRegressor as ET
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.linear_model import LinearRegression as LR
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import gmean
from copy import copy
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor as KNN
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(76352) # Make the keras neural networks reproducible
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD,Adam
from keras.constraints import non_neg

# Global define paths to the files
train_path = "~/Documents/Kaggle/Sberbank/train.csv"
test_path = "~/Documents/Kaggle/Sberbank/test.csv"
macro_path = "~/Documents/Kaggle/Sberbank/macro.csv"


def RMSLE(ypred, ytrue):
    return np.sqrt(1./len(ypred) * np.sum(np.square(np.log(ypred+1) - np.log(ytrue+1))))

def RMSE(ypred, ytrue):
    return np.sqrt(1./len(ypred) * np.sum(np.square(ypred - ytrue)))

class LinearEnsemble:
    '''
    Class to create a linear ensemble object with a scikit-learn like API
    Methods:
    fit(x,y)
        fit the bagger model on feature array x and labels y.

    predict(test)
        Make predictions for new test data based on what the model learned in the fit procedure. 
    '''
    def __init__(self, model_list, weights):
        self.model_list = model_list
        self.weights = weights
        return

    def fit(self,x,y):
        for key,value in self.model_list.items():
            self.model_list[key].fit(x,y)
        return

    def predict(self,test):
        self.model_preds = {}
        self.final_pred = np.zeros(len(test))
        for key, value in self.model_list.items():
            self.model_preds[key] = self.model_list[key].predict(test)
            self.final_pred += self.weights[key] * self.model_preds[key]
        return self.final_pred

class bagger:
    '''
    Class to create a bagger object which does Boostrap Aggregation (Bagging) of any chosen model. 
    The bagger object has an API similar to scikit learn models and can be used in a similar fashion.

    Initialize: 
    bag = bagger(clf,num_bags=100,bag_fraction=0.8)
    where clf is any scikit learn model (example clf = sklearn.ensemble.GradientBoostingClassifier) or an XGC object.
    bag_fraction = decides what percentage of samples must be selected for each bag.
    num_bags = number of bags.
    
    Methods:
    fit(x,y)
        fit the bagger model on feature array x and labels y.

    predict(test)
        Make predictions for new test data based on what the model learned in the fit procedure. 
    '''
    def __init__(self,clf,num_bags=10,bag_fraction=0.75):
        self.clf=clf
        self.num_bags=num_bags
        self.bag_fraction = bag_fraction
        return
    def fit(self,x,y):
        trained_models=[]
        for bag in range(self.num_bags):
            x,y = shuffle(x,y,random_state=bag*3)
            #xtrain,xtest, ytrain,ytest = split(x, y, test_size=1.0-self.bag_fraction, stratify=y,random_state=42*bag)
            xtrain,xtest, ytrain,ytest = train_test_split(x, y, test_size=1.0-self.bag_fraction,random_state=4342+bag*919)
            mod = copy(self.clf)
            mod.fit(xtrain,ytrain)
            trained_models.append(mod)
        self.trained_models=trained_models
    def predict(self,x):
        preds = np.zeros((len(x),self.num_bags))
        for n,mod in enumerate(self.trained_models):
            preds[:,n]=mod.predict(x)
        avg_pred = gmean(preds,axis=1)
        return np.ravel(avg_pred)

def feature_engg(train, test):
    '''
    The mega function to do all the cleaning up
    '''
    train['train_test_ind'] = 0
    test['train_test_ind'] = 1

    train = train[train['full_sq']>= 10]
    test_full_sq = np.array(test['full_sq'])
    test_full_sq[test_full_sq<=10] = 10
    test['full_sq'] = test_full_sq
    '''

    train = train[train['price_doc']/train['full_sq'] < 2000000]

    trainsub = train[train.timestamp < '2015-01-01']
    trainsub = trainsub[trainsub.product_type=="Investment"]

    ind_1m = trainsub[trainsub.price_doc <= 1000000].index
    ind_2m = trainsub[trainsub.price_doc == 2000000].index
    ind_3m = trainsub[trainsub.price_doc == 3000000].index

    train_index = set(train.index.copy())

    for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):
        ind_set = set(ind)
        ind_set_cut = ind.difference(set(ind[::gap]))
        train_index = train_index.difference(ind_set_cut)

    train = train.loc[train_index]
    '''

    x = pd.concat([train,test],axis=0)

    date = np.array(x['timestamp'])
    month = np.array([int(p.split('-')[1]) for p in date])
    year = np.array([int(p.split('-')[0]) for p in date])
    x['month'] = month
    x['year'] = year

    year_str = [str(p) for p in year]
    month_str = [str(p) for p in month]
    year_month = np.array([int(p+q) if int(q)>9 else int(p+'0'+q) for (p,q) in zip(year_str, month_str)])
    x['year_month'] = year_month

    vars_to_encode = ['product_type','sub_area','culture_objects_top_25','thermal_power_plant_raion',
    'incineration_raion','oil_chemistry_raion','radiation_raion','railroad_terminal_raion',
    'big_market_raion','nuclear_reactor_raion','detention_facility_raion','water_1line',
    'big_road1_1line','railroad_1line','ecology','material']
    #x_ohe = pd.get_dummies(x, columns=vars_to_encode)
    x = pd.get_dummies(x,columns=vars_to_encode)
    x['ratio_carpet_to_full_sq'] = x['life_sq']/(x['full_sq']+1)
    x['ratio_area_per_room'] = x['full_sq'] / (x['num_room'] + 1)
    x['ratio_school_children'] = x['children_school']/(x['raion_popul']+1)
    x['ratio_school_quota'] = x['children_school']/(x['school_quota']+1)
    x['ratio_female_to_male'] = x['female_f']/(x['male_f']+1)
    x['ratio_young_female_to_male'] = x['young_female']/(x['young_male']+1)
    x['ratio_ekder'] = x['ekder_all']/(x['full_all']+1)
    x['ratio_working_age'] = x['16_29_all']/(x['full_all']+1)
    x['ratio_old_buildings'] = (x['build_count_1946-1970']+x['build_count_1971-1995'])/(x['raion_build_count_with_builddate_info']+1)
    x['ratio_historic_buildings'] = x['build_count_before_1920']/(x['raion_build_count_with_builddate_info']+1)
    x['ratio_new_buildings'] = x['build_count_after_1995']/(x['raion_build_count_with_builddate_info']+1)
    x['population_density'] = x['raion_popul']/(x['area_m']+1)

    x = x[x['state'] < 5]

    cafe_variables = [p for p in list(x) if 'cafe' in p]
    x_cafe = x[cafe_variables]
    x_cafe = x_cafe.fillna(0)
    pca95 = PCA(n_components=6,whiten=True)
    pca95.fit(x_cafe)
    x_cafe_pcs = pca95.transform(x_cafe)
    x = x.assign(cafe_PC1= x_cafe_pcs[:,0], cafe_PC2=x_cafe_pcs[:,1],
                 cafe_PC3=x_cafe_pcs[:,2], cafe_PC4=x_cafe_pcs[:,3],
                 cafe_PC5=x_cafe_pcs[:,4], cafe_PC6=x_cafe_pcs[:,5])
    x = x.drop(cafe_variables, axis=1)

    office_variables = [p for p in list(x) if 'office' in p]
    x_office = x[office_variables]
    x_office = x_office.fillna(0)
    pca95 = PCA(n_components=1,whiten=True)
    pca95.fit(x_office)
    x_office_pcs = pca95.transform(x_office)
    x = x.assign(office_PC1= np.ravel(x_office_pcs))
    x = x.drop(office_variables, axis=1)

    trc_variables = [p for p in list(x) if 'trc' in p]
    x_trc = x[trc_variables]
    x_trc = x_trc.fillna(0)
    pca95 = PCA(n_components=1,whiten=True)
    pca95.fit(x_trc)
    x_trc_pcs = pca95.transform(x_trc)
    x = x.assign(trc_PC1= x_trc_pcs[:,0], trc_PC2= x_trc_pcs[:,0])
    x = x.drop(trc_variables, axis=1)

    pop_variables = ['full_all','male_f','female_f','young_all','young_male','young_female','work_all',
                'work_male','work_female','ekder_all','ekder_male','ekder_female','0_6_all', 
                '0_6_male','0_6_female','7_14_all','7_14_male','7_14_female', '0_17_all',
                '0_17_male','0_17_female','16_29_all','16_29_male','16_29_female','0_13_all',
                '0_13_male','0_13_female','raion_popul']
    x_pop = x[pop_variables]
    x_pop = x_pop.fillna(0)
    pca95 = PCA(n_components=2,whiten=True)
    pca95.fit(x_pop)
    x_pop_pcs = pca95.transform(x_pop)
    x = x.assign(pop_PC1= x_pop_pcs[:,0], pop_PC2= x_pop_pcs[:,0])

    # Drop all the absolute numbers
    x = x.drop([ #Drop all the absolute numbers. Not needed since relative values have been calculated.
                'full_all','male_f','female_f','young_all','young_male','young_female','work_all',
                'work_male','work_female','ekder_all','ekder_male','ekder_female','0_6_all','0_6_male',
                '0_6_female','7_14_all','7_14_male','7_14_female', '0_17_all','0_17_male','0_17_female',
                '16_29_all','16_29_male','16_29_female','0_13_all','0_13_male','0_13_female',
                'raion_build_count_with_builddate_info','build_count_before_1920','build_count_1921-1945',
                'build_count_1946-1970','build_count_1971-1995','build_count_after_1995','raion_popul',
                'children_school','school_quota','timestamp',
                # Drop the ID features
                'ID_big_road1','ID_big_road2','ID_railroad_terminal',
                'ID_bus_terminal','ID_railroad_station_walk','ID_railroad_station_avto','ID_metro',
                # Drop redundant binary labels. Need only one column to encode them and pandas-dummies makes two.
                # To-do: change dummies to scikit label encoding for binary classes.
                'product_type_OwnerOccupier','culture_objects_top_25_no','thermal_power_plant_raion_no',
                'incineration_raion_no','oil_chemistry_raion_no','radiation_raion_no','railroad_terminal_raion_no',
                'big_market_raion_no','nuclear_reactor_raion_no','detention_facility_raion_no','water_1line_no',
                'big_road1_1line_no','railroad_1line_no'], axis=1)
    y = np.array(x['price_doc'])
    # Dump some bad data points which are screwing up the CV and predictions:
    drop_indices = (x.year_month <= 201208) & (x.product_type_Investment == 0)  # |(xtrain.year_month > 201500)
    keep_indices = np.array([not (p) for p in drop_indices])
    x = x[keep_indices]
    y = y[keep_indices]
    #x = (x - x.mean()) / (x.std())
    x.apply(lambda p: (p - np.mean(p)) / (np.std(p)))
    xtrain,xtest, y = x.loc[x.train_test_ind==0], x.loc[x.train_test_ind==1], y[x.train_test_ind==0]
    test_id = np.array(xtest['id'])
    xtrain,xtest = xtrain.drop(['train_test_ind','year','month','year_month'],axis=1), xtest.drop(['train_test_ind','year','month','year_month'],axis=1)
    #xtrain.to_csv('train_features.csv', index=False)
    #xtest.to_csv('test_features.csv', index=False)
    xtrain,xtest = xtrain.drop(['id'],axis=1), xtest.drop(['id'],axis=1)
    xtrain,xtest = xtrain.drop(['price_doc'],axis=1), xtest.drop(['price_doc'],axis=1)

    return xtrain, y , xtest, test_id

def price_per_sq_ft(x,y):
    area = np.array(x['full_sq'])
    ppsqft = y/1000/area
    return ppsqft

def total_price(x,ppsqft):
    area = np.ravel(x[:,0])
    total_price = ppsqft * 1000 * area
    return total_price

def one_hot(x):
    '''
    # Function to one-hot-encode categorical variables.
    # Arguments: x: DataFrame
    # Returns: x_ohe : DataFrame with OHE'd variables
    '''
    vars_to_encode = ['product_type','sub_area','culture_objects_top_25','thermal_power_plant_raion',
    'incineration_raion','oil_chemistry_raion','radiation_raion','railroad_terminal_raion',
    'big_market_raion','nuclear_reactor_raion','detention_facility_raion','water_1line',
    'big_road1_1line','railroad_1line','ecology']
    #x_ohe = pd.get_dummies(x, columns=vars_to_encode)
    x_ohe = pd.get_dummies(x)
    return x_ohe

def read_macro_data(macro_path):
    macro_file = pd.read_csv(macro_path)
    macro_file = macro_file.drop(['child_on_acc_pre_school', 'modern_education_share', 'old_education_build_share'], axis=1)
    return macro_file

def read_train_data(path, macro_file):
    x = pd.read_csv(path)
    #print(x.head(5))
    #train = pd.merge(x,macro_file,how='left', on='timestamp')
    train = x
    train = train.loc[train.full_sq < 410]
    #print(train.head(5))
    xtrain = train.fillna(-2)

    #xtrain = xtrain[best_vars]
    #xtrain = xtrain.fillna(0)
    return xtrain

def read_test_data(path, macro_file):
    test = pd.read_csv(path)
    test['price_doc'] = 0
    #print(x.head(5))
    #test = pd.merge(test,macro_file,how='left', on='timestamp')
    xtest = test.fillna(-2)

    #xtest = xtest.fillna(xtest.mean())
    return xtest

def cross_validate(model, x, y, folds=5, runs=1):
    ypred = np.zeros((len(y),runs))
    fold_rmsle = np.zeros((runs,folds))
    r=0
    score = np.zeros(runs)
    col_names = list(x)
    ppsqft = price_per_sq_ft(x,y)
    x = np.array(x)
    y = ppsqft
    offset=2000
    #y = np.log(y+1) # Convert to logs as the cost function is RMSLE and XGBoost uses RMSE by default
    for run in range(runs):
        i=0
        x,y = shuffle(x,y,random_state=19*(run+3))
        kf = KFold(n_splits=folds,random_state=17*(run+93))
        print('Cross Validating...')
        for train_ind,test_ind in kf.split(x):
            print('CV Fold',str(i+1),'out of',str(folds))
            xtrain,ytrain = x[train_ind,:],y[train_ind]
            xtest,ytest = x[test_ind,:],y[test_ind]
            model.fit(xtrain,ytrain)
            fold_pred = model.predict(xtest)
            fold_pred[fold_pred < 0] = 0
            fold_rmsle[r,i] = RMSLE(total_price(xtest,fold_pred), total_price(xtest,ytest))
            #fold_rmsle[r,i] = RMSLE(fold_pred, ytest)
            ypred[test_ind,r]=fold_pred
            i=i+1
        score[r] = RMSLE(total_price(x,ypred[:,r]),total_price(x,y))
        #score[r] = RMSLE(y,ypred[:,r])
        r=r+1
    print('Fold RMSLE:', str(fold_rmsle))
    print('Mean:',str(np.mean(fold_rmsle)))
    print('Deviation:',str(np.std(fold_rmsle)))
    # Feature importances - Does not work for ensembles
    #imps = np.ravel(model.feature_importances_)
    #imp_table = pd.DataFrame({'Variable':col_names, 'Importance':imps})
    #imp_table.to_csv('feature_importance.csv', index=False)
    print('\nOverall RMSLE:',str(score))
    print('Mean:',str(np.mean(score)))
    print('Deviation:',str(np.std(score)))
    return score

def Blender(model_list, data, labels, holdout_fraction = 0.2, repeats = 1):
    """
    WARNING - POTENTIAL DATA LEAK / OVERFITTING!!
    Fit base models on training data and estimate best blend on hold-outs. Repeat and average using different random seeds. Uses Lasso.
    :param dict model_list: dictionary of models to be blended. Models need to be scikit or XGB. No Deep learning at the moment :(
    :param array data: training data
    :param array labels: training labels
    :param float holdout_fraction: fraction of training data to be left out for lv2
    :param int repeats: how many times to average the calculated coefficients
    :return dict blend_coefs: Dictionary with models and their best linear combination coefficients
    """
    model_names = list(model_list)
    coef_table = pd.DataFrame(data = np.zeros([repeats, len(model_names)]), columns = model_names)
    R2_table = pd.DataFrame(data=np.zeros([repeats, len(model_names)]), columns=model_names)
    RMSE_table = pd.DataFrame(data=np.zeros([repeats, len(model_names)]), columns=model_names)
    for r in range(repeats):
        print("Repeat",(r+1),"out of",repeats)
        xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size = holdout_fraction, random_state = 49*(r+97))
        test_oobs = pd.DataFrame(data = np.zeros([len(xtest), len(model_names)]), columns = model_names)
        for key, model in model_list.items():
            reg = copy(model) # just to ensure we're not referencing by memory. Python is arbitrary.
            reg.fit(xtrain, ytrain)
            test_oobs[key] = reg.predict(xtest)
            R2_table.loc[r,key] = RMSLE(test_oobs[key], ytest)
            RMSE_table.loc[r, key] = RMSE(test_oobs[key], ytest)

        L1 = Lasso(alpha = 0.001, fit_intercept = False, precompute = True, max_iter = 800, tol = 0.0001, positive = True, random_state = 149*(r+7), selection = 'random')
        L2 = Ridge(alpha = 5.0)
        lin = LR()
        regressor = lin
        lin.fit(np.array(test_oobs), ytest)
        coef_table.loc[r] = regressor.coef_
    print(coef_table)
    print(R2_table)
    print(RMSE_table)

def StackOOBMaker(train, y, test, base_models, folds=5, repeats=1):
    train, test = np.array(train), np.array(test)
    train_oobs = pd.DataFrame(data=np.zeros([len(y), len(base_models)]), columns=list(base_models))
    for r in range(repeats):
        kf = KFold(n_splits=folds,random_state=(r+37)*91)
        i=0
        print('Making train OOBs repeat',r+1,'out of', repeats)
        for train_ind,test_ind in kf.split(train):
            print('Partition OOB:',str(i+1),'out of',str(folds))
            xtrain,ytrain = train[train_ind,:],y[train_ind]
            xtest,_ = train[test_ind,:],y[test_ind]
            for key, model in base_models.items():
                reg = copy(model)
                reg.fit(xtrain,ytrain)
                train_oobs.loc[test_ind,key] += (1.0/repeats) * reg.predict(xtest)
            i+=1
    train_oobs.to_csv('train_oobs.csv', index=False)
    print('Making test OOBs')
    test_oobs = pd.DataFrame(data=np.zeros([len(test),len(base_models)]), columns=list(base_models))
    for key, model in base_models.items():
        reg = copy(model)
        reg.fit(train,y)
        test_oobs[key] = reg.predict(test)
    test_oobs.to_csv('test_oobs.csv', index=False)
    return

def train_and_predict(model, train, y, test, test_id, submission_file_name = 'xgb_sub1.csv'):
    y2 = price_per_sq_ft(train,y)
    xtrain,xtest = np.array(train), np.array(test)
    model.fit(xtrain,y2)
    ypred = model.predict(xtest)
    ypred2 = total_price(xtest,ypred)
    sub_df = pd.DataFrame({'id':test_id, 'price_doc': ypred2})
    sub_df.to_csv(submission_file_name, index=False)
    return

def NeuralNetModel():
    # create model
    input_feats = 333#-1 # The -1 is to account for ID, which is dropped for NN but is kept for XGb
    hidden_layer_nodes = 10
    h1 = hidden_layer_nodes
    h2 = hidden_layer_nodes
    h3 = hidden_layer_nodes
    model = Sequential()
    model.add(Dropout(0.10, input_shape=(input_feats,)))
    #model.add(Dense(units=h1, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=h1, kernel_constraint = non_neg(), activation='relu'))
    #model.add(Dropout(0.20))
    #model.add(Dense(units=h2, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(units=h2, kernel_constraint=non_neg(), activation='relu'))
    #model.add(Dropout(0.20))
    #model.add(Dense(units=h3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=h3, kernel_constraint=non_neg(), activation='relu'))
    #model.add(Dropout(0.20))
    #model.add(Dense(units=1, kernel_initializer='normal'))
    model.add(Dense(units=1, kernel_constraint=non_neg()))
    # Compile model
    #sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    #model.compile(loss='mean_squared_error', optimizer=sgd)
    adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model


def main():
    print('Reading files...')
    macro_file = read_macro_data(macro_path)
    train = read_train_data(train_path, macro_file)
    test = read_test_data(test_path, macro_file)
    xtrain, y , xtest, test_id = feature_engg(train, test)
    print(np.shape(xtrain))
    #xtrain = train
    #xtrain['y']=y
    #xtrain.to_csv('train_features.csv', index=False)
    lin = LR()


    xgbm = xgb.XGBRegressor(max_depth=6, learning_rate=0.01, n_estimators=2700, silent=True,
                            objective='reg:linear', nthread=-1, gamma=0, min_child_weight=1,
                            max_delta_step=0, subsample=0.9, colsample_bytree=0.7, colsample_bylevel=1,
                            reg_alpha=20, reg_lambda=20, scale_pos_weight=1, base_score=150, seed=4635,
                            missing=None)

    xgbm2 = xgb.XGBRegressor(max_depth=6, learning_rate=0.2, n_estimators=200, silent=True,
                            objective='reg:linear', nthread=-1, gamma=0, min_child_weight=1,
                            max_delta_step=0, subsample=0.7, colsample_bytree=0.7, colsample_bylevel=1,
                            reg_alpha=20, reg_lambda=20, scale_pos_weight=1, base_score=150, seed=4635,
                            missing=None)

    '''
    xgbm = xgb.XGBRegressor(max_depth=6, learning_rate=0.02, n_estimators=700, silent=True, 
                            objective='reg:linear', nthread=-1, gamma=0, min_child_weight=0.1, 
                            max_delta_step=0, subsample=0.7, colsample_bytree=0.7, colsample_bylevel=1, 
                            reg_alpha=10, reg_lambda=10, scale_pos_weight=1, base_score=135, seed=4635, 
                            missing=None)
    '''
    extra = ET(n_estimators=500, criterion='mse', max_depth=20, min_samples_split=4, min_samples_leaf=2,
                min_weight_fraction_leaf=0.0, max_features=80, max_leaf_nodes=None, min_impurity_split=1e-07,
                bootstrap=False, oob_score=False, n_jobs=8, random_state=0, verbose=0, warm_start=False)

    forest = RF(n_estimators=1000, criterion='mse', max_depth=15, min_samples_split=4, min_samples_leaf=2,
                min_weight_fraction_leaf=0.0, max_features=80, max_leaf_nodes=None, min_impurity_split=1e-07,
                bootstrap=False, oob_score=False, n_jobs=8, random_state=0, verbose=0, warm_start=False)

    L1 = Lasso(alpha=0.001, fit_intercept=True, normalize=True, precompute=True, copy_X=True,
               max_iter=1000, tol=0.001, warm_start=False, positive=False,random_state=99, selection='random')

    L2 = Ridge(alpha=5.0)



    neighbors15 = KNN(n_neighbors=15, weights='uniform', algorithm='kd_tree', leaf_size=40,
                      p=2, metric='minkowski', metric_params=None, n_jobs=-1)

    neighbors5 = KNN(n_neighbors=5, weights='uniform', algorithm='kd_tree', leaf_size=40,
                      p=2, metric='minkowski', metric_params=None, n_jobs=-1)

    neighbors25 = KNN(n_neighbors=25, weights='uniform', algorithm='kd_tree', leaf_size=40,
                     p=2, metric='minkowski', metric_params=None, n_jobs=-1)

    deepnet = KerasRegressor(build_fn=NeuralNetModel, epochs=80, verbose=0)

    ens = LinearEnsemble(model_list = {'xgbm':xgbm,'forest':forest}, weights = {'xgbm':0.80,'forest':0.18})
    neighborhood = LinearEnsemble(model_list={'KNN15': neighbors15, 'KNN5': neighbors5, 'KNN25': neighbors25},
                                  weights={'KNN15': 0.5, 'KNN5': 0.25, 'KNN25':0.22})
    xgb_bag = bagger(xgbm,num_bags=50,bag_fraction=0.85)
    ens_xgb_knn15 = LinearEnsemble(model_list={'xgbm': xgbm, 'KNN15': neighbors15}, weights={'xgbm': 0.85, 'KNN15': 0.12})
    ens_xgb__rf_knn5 = LinearEnsemble(model_list={'xgbm': xgbm, 'RF':forest, 'KNN5': neighbors5}, weights={'xgbm': 0.8, 'RF': 0.12, 'KNN5': 0.04})

    #cross_validate(ens_xgb__rf_knn15, xtrain, y, folds=5, runs=1)

    models_for_stack = {'XGB':xgbm, 'RF':forest,
                        #'Lasso':L1,
                        #'Ridge': L2,
                        'ETC': extra, 'KNN5': neighbors5, 'KNN15': neighbors15,
                        'KNN25': neighbors25}
    #Blender(models_for_stack, xtrain, y, holdout_fraction=0.2, repeats=1)
    #StackOOBMaker(xtrain, y, xtest, base_models=models_for_stack, folds=5)
    #train.to_csv('train_merged.csv', index = False)
    train_and_predict(bagger(ens_xgb__rf_knn5,num_bags=100,bag_fraction=0.9), xtrain, y, xtest, test_id, submission_file_name = 'xgb_rf_knn5_sub_7x.csv')
    '''
    train_oobs = pd.read_csv('train_oobs.csv')
    test_model_corrs = train_oobs.corr()
    sns.heatmap(test_model_corrs, vmax=1.0, square=True, cmap="Blues")
    plt.title("Model Correlation Heatmap", fontsize=18)
    plt.show()
    '''

    return

if __name__ == '__main__':
    main()

