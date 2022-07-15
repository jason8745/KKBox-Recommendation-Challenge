from sklearn import model_selection, metrics, ensemble
import pandas as pd
from utils.time_func import timer
import xgboost as xgb

class Trainer:
    def __init__(self) -> None:
        pass
    @timer
    def __get_importance_matrix_from_random_forest(self,train)->pd.DataFrame:
        """
        透過randomforest初步篩出重要的變數
        """
        model = ensemble.RandomForestClassifier(n_estimators=250, max_depth=25)
        model.fit(train[train.columns[train.columns != 'target']], train.target)
        importance_matrix = pd.DataFrame({'features': train.columns[train.columns != 'target'],
                    'importances': model.feature_importances_})
        return importance_matrix
    @timer
    def __drop_variable_under_threshold(self,train,importance_matrix)->pd.DataFrame:
        train = train.drop(importance_matrix.features[importance_matrix.importances < 0.04].tolist(), 1)
        return train

    @timer
    def __split_target_label(self,train):
        """
        pop is inplace change
        """
        target = train.pop('target')
        return train,target


    def split_dataset_to_train_and_val(self,train:pd.DataFrame)->dict:
        importance_matrix = self.__get_importance_matrix_from_random_forest(train)
        train = self.__drop_variable_under_threshold(train = train,importance_matrix=importance_matrix)
        train,target = self.__split_target_label(train)
        train_data, test_data, train_labels, test_labels = model_selection.train_test_split(train, target, test_size = 0.2)
        return {'train_data':train_data,
                'test_data': test_data, 
                'train_labels':train_labels,
                'test_labels': test_labels
                }

    @timer
    def xgboost_grid_search_find_best_param(self, dataset_dict: dict,param_grid: dict ):
        model = xgb.XGBClassifier()
        cv = model_selection.StratifiedShuffleSplit(test_size = 0.2)
        grid_cv = model_selection.RandomizedSearchCV(model, param_grid, scoring = 'accuracy', cv = cv)
        grid_cv.fit(dataset_dict['train_data'], dataset_dict['train_labels'])
        return grid_cv.best_estimator_








        



    

    




        



        




