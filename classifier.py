#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score, classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import pandas as pd
import pickle


# In[ ]:


def load_data(file_name):
    """
    Load data from a file using Pickle
    file_name: the name of the file needed to be loaded
    Return: the data in the form before saved
    """
    with open(file_name,'rb') as f:
        X = pickle.load(f)
    return X


# In[4]:


def classify(X_train_file, y_train, X_test_file, y_test, feature = "stylometric"):
    """
    Train three different machine learning algorithms with each type of feature vector and report on the best results
    First the data are scaled using Standard Scaler then fed to each algorithm.
    Using grid search for parameter optimization
    X_train_file: the path to the file containg the train vectors
    y_train: a list of train labels
    X_test_file: the path to the file containg the test vectors
    y_test: a list of test labels
    feature: name of the feature set
        
    Return a message
    
    References:
    Documentation for optimisation in sklearn library
    https://scikit-learn.org/stable/modules/grid_search.html
    Documentation for SGD classifier in sklearn library
    https://scikit-learn.org/stable/modules/sgd.html#stochastic-gradient-descent
    Pipeline tutorials:
    https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976
    """
    pl_lr = Pipeline([("scaler", StandardScaler()),
                      ("LR", LogisticRegression(random_state=30, max_iter=10000))])
    
    pl_rf = Pipeline([("scaler", StandardScaler()),
                      ("RF", RandomForestClassifier(random_state=30))])
    
    pl_sgd = Pipeline([("scaler", StandardScaler()),
                       ("SGD", SGDClassifier(class_weight= "balanced", random_state=30, max_iter=10000))])

    prm_lr = {"LR__C": [10e-2,0.1,10,100],
              "LR__solver": ["liblinear", "lbfgs"]}

    prm_rf = [{"RF__n_estimators" : [50 ,100, 150],
               "RF__criterion" : ["gini", "entropy"],
               "RF__max_features" : ["sqrt", "log2"],
               "RF__max_depth" : [2, 4, 6]}]
    
    prm_sgd = [{"SGD__loss": ['hinge','log','modified_huber','squared_hinge','perceptron'],
                "SGD__penalty": ['l2', 'l1'],
                "SGD__learning_rate" : ["constant","optimal","invscaling","adaptive"],
                "SGD__eta0":[0.001,0.01, 0.1, 1],
                "SGD__tol": [10e-4, 10e-3, 10e-2]}]

    jobs = -1

    gs_lr = GridSearchCV(estimator = pl_lr,
                         param_grid = prm_lr,
                         cv = 3,
                         n_jobs = jobs)

    gs_rf = GridSearchCV(estimator = pl_rf,
                         param_grid = prm_rf,
                         cv = 3,
                         n_jobs = jobs)
    
    gs_sgd = GridSearchCV(estimator = pl_sgd,
                          param_grid = prm_sgd,
                          cv = 3,
                          n_jobs = jobs)

    clfs = [gs_lr, gs_rf, gs_sgd]

    # Dictionary of pipelines and classifier types for ease of reference
    clf_dict = {0:"Logistic Regression", 1:"Random Forest", 2:"Stochastic Gradient Descent"}

    #Load the train and test data from files
    X_train_vecs = load_data(X_train_file)
    X_test_vecs = load_data(X_test_file)
    
    results = []
    #Fit each grid search with the data
    for index, clf in tqdm(enumerate(clfs)):
        estimator = clf_dict[index] #match the name with the classifier
        print(f"\nClassifier:{estimator}")
        # Fit grid search
        print("fitting the grid search...")
        clf.fit(X_train_vecs, y_train)
        # Best parameters
        best_prm = clf.best_params_
        print(f"Best params: {best_prm}")
        # Predict on test data with best parameters
        y_pred = clf.predict(X_test_vecs)
        with open ("Results/"+estimator+"_"+feature+".txt", "w", encoding = "utf-8") as infile:
            infile.write("\n".join(y_pred))
        #Evaluate the performance of the classifier on the test set
        #Accuracy score
        accuracy = accuracy_score(y_test,y_pred)
        print(f"Accuracy:{accuracy}")
        #Classification report
        clf_report = classification_report(y_test,y_pred,labels=['male','female'])
        #Confusion matrix
        cfm = confusion_matrix(y_test, y_pred, labels=['male','female'])
        cm_df = pd.DataFrame(cfm , index = ['male','female'], columns = ['male','female'])
        result = f'\nClassifier:{estimator}\nBest prarameters:{best_prm}\nAccuracy:{accuracy}\nClassification report:{clf_report}\n{cm_df}'
        results.append(result)
    
    save_file_name = "Results/"+ feature +"_results.txt"

    #Write down the result in the save file
    with open (save_file_name,'w', encoding = "utf-8") as f:
        f.write("\n".join(results))
            
    return f"For a more detailed report on the result, open {save_file_name}"
