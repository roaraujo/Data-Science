
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, auc, roc_curve, precision_score, recall_score, classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, train_test_split

def holdout_clf(clf, X, Y, test_s=0.3, strat = None):
    """_summary_

    Args:
        clf (_model_): _model_
        X (_dataframe_): _var pred_
        Y (_array_): _var id_
        test_s (float, optional): _description_. Defaults to 0.3.
        strat (_type_, optional): _description_. Defaults to None.
    """
    # ----------- Hold out ----------- #
    sc_mean_test=list()
    sc_mean_train=list()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_s, random_state=42, stratify=strat)

    clf = clf.fit(X_train, y_train)

    predicted_train = clf.predict(X_train)
    sc_mean_train = accuracy_score(y_train, predicted_train)

    predicted_test = clf.predict(X_test)
    sc_mean_test= accuracy_score(y_test, predicted_test)

    print("Média da Accuracy base de treinamento: ",sc_mean_train)
    print("Média da Accuracy base de teste: ", sc_mean_test)

    print("Matriz de confusão base de treinamento: \n", confusion_matrix(y_train, predicted_train))  
    print("Classification Report base de treinamento: \n", classification_report(y_train, predicted_train)) 

    print("Matriz de confusão base de teste: \n", confusion_matrix(y_test, predicted_test))  
    print("Classification Report base de teste: \n", classification_report(y_test, predicted_test)) 
    
    return clf 

def leave_one_out_clf(clf, X, Y):
    # ----------- Leave-one-out ----------- #
    sc_mean_test= list()
    sc_mean_train= list()
    LL_test=list()
    LL_train=list()

    cv = LeaveOneOut() # 
    for train_index, test_index in cv.split(X.values):
        X_train, X_test = X.values[train_index,:], X.values[test_index,:]
        y_train, y_test = Y[train_index],Y[test_index]

        y_test = np.array(y_test)   
        y_train = np.array(y_train)        
            
        clf = clf.fit(X_train, y_train)
        predicted_train = clf.predict(X_train)
        sc_mean_train.append(accuracy_score(y_train, predicted_train))
        
        predicted_test = clf.predict(X_test)
        sc_mean_test.append(accuracy_score(y_test, predicted_test))


    print("Média da Accuracy base de treinamento: ", np.mean(sc_mean_train))
    print("Média da Accuracy base de teste: ", np.mean(sc_mean_test))

    return clf 


def kfold_clf(clf, X, Y, k=10):
    # ----------- K-fold Cross Validation ----------- #
    sc_mean_test=list()
    sc_mean_train=list()

    LL_test=list()
    LL_train=list()

    cv = KFold(n_splits=k) 
    for train_index, test_index in cv.split(X.values):
        X_train, X_test = X.values[train_index,:], X.values[test_index,:]
        y_train, y_test = Y[train_index],Y[test_index]  

        y_test = np.array(y_test)   
        y_train = np.array(y_train)    

        clf = clf.fit(X_train, y_train)
        predicted_train = clf.predict(X_train)
        sc_mean_train.append(accuracy_score(y_train, predicted_train))
        
        predicted_test = clf.predict(X_test)
        sc_mean_test.append(accuracy_score(y_test, predicted_test))

        LL_test.append(log_loss(y_test, predicted_test))
        LL_train.append(log_loss(y_train, predicted_train))


    print("Média da Accuracy base de treinamento: ", np.mean(sc_mean_train))
    print("Média da Accuracy base de teste: ", np.mean(sc_mean_test))

    print("Média do LogLoss base de validação: ", np.mean(LL_test))
    print("Média do LogLoss base de treinamento: ", np.mean(LL_train))

    return clf 

# Kfolds / Leave one out para resposta com 2 níveis

def model_classif_cv(model, X, y, cv, metrics):

    first = True

    for train_index, test_index in cv.split(X):
        #print(train_index)
        model2 = model.fit(X.iloc[train_index],y[train_index])
        
        pred_train = model2.predict(X.iloc[train_index])
        pred_test = model2.predict(X.iloc[test_index])
        
        prob_train = model2.predict_proba(X.iloc[train_index])
        prob_test = model2.predict_proba(X.iloc[test_index])

        prob1_train = pd.DataFrame(prob_train).iloc[:,1]
        prob1_test = pd.DataFrame(prob_test).iloc[:,1]

        prob0_train = pd.DataFrame(prob_train).iloc[:,0]
        prob0_test = pd.DataFrame(prob_test).iloc[:,0]
        
        y_train = y[train_index]
        y_test = y[test_index]
        
        train_results = pd.concat([y_train.reset_index(drop=True), prob1_train], axis = 1)
        test_results = pd.concat([y_test.reset_index(drop=True), prob1_test], axis = 1)
        train_results.columns = ['y_train', 'prob1']
        test_results.columns = ['y_test', 'prob1']
            
        first_metric = True
        for metric in metrics:
            name_metric = metric.__name__
            
            if metric == roc_auc_score:
                m_tr = metric(y_train, prob1_train)
                m_te = metric(y_test, prob1_test)
                
            else:
                m_tr = metric(y_train, pred_train)
                m_te = metric(y_test, pred_test)
            
            m_tr_te = {
                str(name_metric) +'- 1.Treino': [m_tr],
                str(name_metric) +'- 2.Teste' : [m_te]
            }
            resultados_aux = pd.DataFrame(m_tr_te)
            
            if first_metric == True:
                results_folds = resultados_aux
                first_metric = False
            else:
                results_folds = pd.concat([results_folds, resultados_aux], axis = 1)    
                
        train_prob1_True1 = train_results[train_results.y_train == 1][['prob1']]
        train_prob1_True0 = train_results[train_results.y_train == 0][['prob1']]
        
        test_prob1_True1 = test_results[test_results.y_test == 1][['prob1']]
        test_prob1_True0 = test_results[test_results.y_test == 0][['prob1']]
        
        m_tr_te_ks = {
                    'ks - 1.Treino': [stats.ks_2samp(train_prob1_True1.prob1, train_prob1_True0.prob1).statistic],
                    'ks - 2.Teste' : [stats.ks_2samp(test_prob1_True1.prob1, test_prob1_True0.prob1).statistic]
        }  
                       
        resultados_aux = pd.DataFrame(m_tr_te_ks)
        results_folds = pd.concat([results_folds, resultados_aux], axis = 1)    
    
        if first == True:
            results = results_folds
            first = False
        else:
            results = pd.concat([results, results_folds], axis = 0)    

    results.index = range(cv.get_n_splits(X))
    results_mean = np.transpose(pd.DataFrame(results.mean(), columns=['mean']))
    results = pd.concat([results, results_mean], axis = 0)

    return results


# Validação holdout resposta binária
def model_classif_holdout(clf, X_train, y_train, X_test, y_test, metrics):
    
    clf2 = clf.fit(X_train, y_train)
       
    pred_train = clf2.predict(X_train)
    pred_test = clf2.predict(X_test)

    prob_train = clf2.predict_proba(X_train)
    prob_test = clf2.predict_proba(X_test)

    prob1_train = pd.DataFrame(prob_train).iloc[:,1]
    prob1_test = pd.DataFrame(prob_test).iloc[:,1]
    
    prob0_train = pd.DataFrame(prob_train).iloc[:,0]
    prob0_test = pd.DataFrame(prob_test).iloc[:,0]
    
    train_results = pd.concat([y_train, prob1_train], axis = 1)
    test_results = pd.concat([y_test, prob1_test], axis = 1)
    train_results.columns = ['y_train', 'prob1']
    test_results.columns = ['y_test', 'prob1']
    
    first_metric = True
    
    for metric in metrics:
            name_metric = metric.__name__
            
            if metric == roc_auc_score:
                m_tr = metric(y_train, prob1_train)
                m_te = metric(y_test, prob1_test)

            else:
                m_tr = metric(y_train, pred_train)
                m_te = metric(y_test, pred_test)
            
            m_tr_te = {
                '1.Treino': [m_tr],
                '2.Teste' : [m_te]
            }
            
            resultados_aux = pd.DataFrame(m_tr_te, index = [str(name_metric)])
            #print(resultados_aux)
            if first_metric == True:
                results_folds = resultados_aux
                first_metric = False
            else:
                results_folds = pd.concat([results_folds, resultados_aux], axis = 0)    
    
    train_prob1_True1 = train_results[train_results.y_train == 1][['prob1']]
    train_prob1_True0 = train_results[train_results.y_train == 0][['prob1']]
    
    test_prob1_True1 = test_results[test_results.y_test == 1][['prob1']]
    test_prob1_True0 = test_results[test_results.y_test == 0][['prob1']]
    
    
    m_tr_te_ks = {
                '1.Treino': [stats.ks_2samp(train_prob1_True1.prob1, train_prob1_True0.prob1).statistic],
                '2.Teste' : [stats.ks_2samp(test_prob1_True1.prob1, test_prob1_True0.prob1).statistic]
    }
    
    
    resultados_aux = pd.DataFrame(m_tr_te_ks, index = ['KS'])
    results_folds = pd.concat([results_folds, resultados_aux], axis = 0)    
    
    
    return results_folds