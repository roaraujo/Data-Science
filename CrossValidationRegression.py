
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, train_test_split


def kfold_reg(LM, X, Y, k=10):
    """_summary_

    Args:
        LM (_type_): _description_
        X (_type_): _description_
        Y (_type_): _description_
        k (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    cv = KFold(k)

    # metrics = [mean_squared_error, r2_score, mean_absolute_error]

    first = True

    for train_index, test_index in cv.split(X):
        model = LM.fit(X.iloc[train_index],Y[train_index])
        
        pred_train = model.predict(X.iloc[train_index])
        pred_test = model.predict(X.iloc[test_index])

        y_train = Y[train_index]
        y_test = Y[test_index]
            
        mse_treino = mean_squared_error(y_train, pred_train, squared = True)
        mse_teste = mean_squared_error(y_test, pred_test, squared = True)

        rmse_treino = mean_squared_error(y_train, pred_train, squared = False)
        rmse_teste = mean_squared_error(y_test, pred_test, squared = False)

        mae_treino = mean_absolute_error(y_train, pred_train)
        mae_teste = mean_absolute_error(y_test, pred_test)
            
        r2_treino = r2_score(y_train, pred_train)
        r2_teste = r2_score(y_test, pred_test)
        
        resultados_aux = {
        "MSE: 1. Treino" : [mse_treino] ,
        "MSE: 2. Teste" : [mse_teste],
        "RMSE: 1. Treino" : [rmse_treino] ,
        "RMSE: 2. Teste" : [rmse_teste],
        "MAE: 1. Treino" : [mae_treino],
        "MAE: 2. Teste" : [mae_teste],
        "R2: 1. Treino" : [r2_treino],
        "R2: 2. Teste" : [r2_teste]
        }

        resultados_aux = pd.DataFrame(resultados_aux)

        if first == True:
            results_folds = resultados_aux
            first = False
        else:
            results_folds = pd.concat([results_folds, resultados_aux], axis = 0)

    results_folds.index = range(0,k)
    results_mean = np.transpose(pd.DataFrame(results_folds.mean(), columns=['Média']))
        
    results_folds = pd.concat([results_folds, results_mean], axis = 0)

    return results_folds.round(2)


def leave_one_out_reg(LM, X, Y):
    """_summary_

    Args:
        LM (_type_): _description_
        X (_type_): _description_
        Y (_type_): _description_
    """
    X = np.array(X)
    cv = LeaveOneOut()
    cv.get_n_splits(X)
    mse_train = list()
    mse_teste = list()
    rmse_train = list()
    rmse_teste = list()

    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        LM = LM.fit(X_train,y_train)
        y_predito_train = LM.predict(X_train)
        y_predito_teste = LM.predict(X_test)
        mse_train.append(mean_squared_error(list(y_train), y_predito_train, squared = True))
        mse_teste.append(mean_squared_error(list(y_test), y_predito_teste, squared = True))
        rmse_train.append(mean_squared_error(list(y_train), y_predito_train, squared = False))
        rmse_teste.append(mean_squared_error(list(y_test), y_predito_teste, squared = False))
        
    print("O MSE da base de treino: ", np.array(mse_train).mean())
    print("O MSE da base de teste: ", np.array(mse_teste).mean())
    print("O RMSE da base de treino: ", np.array(rmse_train).mean())
    print("O RMSE da base de teste: ", np.array(rmse_teste).mean())


def th_reg(LM, X, Y, corte = 253):
    """_summary_

    Args:
        LM (_type_): _description_
        X (_type_): _description_
        Y (_type_): _description_
        corte (int, optional): _selecionar ponto de corte na base_. Defaults to 253.
    """
    corte = 253
    c = corte-1
    n = len(X)
    # criando treino e teste 

    X_train = X.iloc[0:corte,:]
    y_train = Y.loc[0:c]
    X_test = X.loc[corte:n,:]
    y_test = Y.iloc[corte:n]

    LM.fit(X_train, y_train)
    pred_train = LM.predict(X_train)
    pred_test = LM.predict(X_test)    

    mse_treino = mean_squared_error(y_train, pred_train, squared = True)
    mse_teste = mean_squared_error(y_test, pred_test, squared = True)

    rmse_treino = mean_squared_error(y_train, pred_train, squared = False)
    rmse_teste = mean_squared_error(y_test, pred_test, squared = False)
            
    mae_treino = mean_absolute_error(y_train, pred_train)
    mae_teste = mean_absolute_error(y_test, pred_test)
            
    R2_treino = r2_score (y_train, pred_train)
    R2_teste = r2_score (y_test, pred_test)

    print("MSE treino:",mse_treino.round(2))
    print("MSE teste:",mse_teste.round(2))
    print("RMSE treino:",rmse_treino.round(2))
    print("RMSE teste:",rmse_teste.round(2))
    print("MAE treino:",mae_treino.round(2))
    print("MAE treino:",mae_teste.round(2))
    print("R2 treino:",R2_treino.round(2))
    print("R2 treino:",R2_teste.round(2))