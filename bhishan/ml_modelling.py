__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various model evaluation utilities.

- print_df_eval
- multiple_linear_regression
- get_binary_classification_scalar_metrics
- get_binary_classification_scalar_metrics2
- get_binary_classification_report
- print_confusion_matrix
- get_false_negative_frauds
- plot_confusion_matrix_plotly
- plot_auc
- plot_roc_skf

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.ml_model_eval?

"""

__all__ = [
    "print_df_eval",
    "adjustedR2",
    "multiple_linear_regression",
    "get_binary_classification_scalar_metrics",
    "get_binary_classification_scalar_metrics2",
    "get_binary_classification_report",
    "print_confusion_matrix",
    "get_false_negative_frauds",
    "plot_confusion_matrix_plotly",
    "plot_roc_auc",
    "plot_roc_skf"
    ]

# type hints
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from typing import Optional, Sequence, Type, TypeVar
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame, Series
from pandas.io.formats.style import Styler
try:
    from .mytyping import (IN, SN, SI, SIN, TL, LD, TLN, LDN,
    DS, DSt, NUM, NUMN, AD, AS, DN,
    ARR, ARRN, SARR, SARRN, LIMIT, LIMITN,
    LTii,LTss,LTff,LTsi,
    )
except:
    from mytyping import (IN, SN, SI, SIN, TL, LD, TLN, LDN,
    DS, DSt, NUM, NUMN, AD, AS, DN,
    ARR, ARRN, SARR, SARRN, LIMIT, LIMITN,
    LTii,LTss,LTff,LTsi,
    )

from IPython.display import display
import matplotlib.pyplot as plt

def print_df_eval():
    """Print a data frame to store the model evalution.
    """
    ans = """df_eval = pd.DataFrame({'Model': [],
                        'Details':[],
                        'Root Mean Squared Error (RMSE)':[],
                        'R-squared (training)':[],
                        'Adjusted R-squared (training)':[],
                        'R-squared (test)':[],
                        'Adjusted R-squared (test)':[],
                        '5-Fold Cross Validation':[]})"""
    print(ans)

def adjustedR2(r2:NUM,
    nrows:int,
    kcols:int
    )-> NUM:
    """Function to calculate adjusted R-squared.

    R-squared metric increases with number of features used.
    To get more robust model comparison metric, we can use
    adjusted R-squared metric which penalizes larger number of featrues.

    (R-adj)^2 = R^2 - (k-1)/(n-k) * (1-R^2)

    Parameters
    ----------
    r2: float
        The r-squared value.
    nrows: int
        Total number of observations.
    kcols: int
        Total number of parameters.

    """
    return r2-(kcols-1)/(nrows-kcols)*(1-r2)

def multiple_linear_regression(
    df: DataFrame,
    features: Iterable,
    target:str,
    model:Any,
    cv:int=5
    )->Tuple:
    """ Multiple Linear Regression Modelling using given model.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe.
    features: list
        list of feature names
    target: string
        target name.
    model: model
        sklearn model e.g. sklearn.ensemble.RandomForestRegressor
    cv: int
        cross-validation

    """
    # train test values
    X = df[features].values
    y = df[target].values.reshape(-1,1)

    Xtrain = train[features].values
    ytrain = train[target].values.reshape(-1,1)

    Xtest = test[features].values
    ytest = test[target].values.reshape(-1,1)

    # random forest regressor wants 0d scalar not 1d vector
    if isinstance(model, sklearn.ensemble.RandomForestRegressor):
        ytrain = train[target].values.ravel()

    # fitting
    model.fit(Xtrain,ytrain)

    # prediction
    ypreds = model.predict(Xtest)

    # metrics
    rmse = np.sqrt(mean_squared_error(ytest,ypreds)).round(3)
    r2_train = model.score(Xtrain, ytrain).round(3)
    r2_test = model.score(Xtest, ytest).round(3)

    cv = cross_val_score(model, X, y, cv=5,n_jobs=-1,verbose=1).mean().round(3)

    ar2_train = adjustedR2(model.score(Xtrain,ytrain),
                            Xtrain.shape[0],
                            len(features)).round(3)
    ar2_test  = adjustedR2(model.score(Xtest,ytest),
                            Xtest.shape[0] ,
                            len(features)).round(3)

    return (rmse, r2_train, ar2_train, r2_test, ar2_test, cv)

def get_binary_classification_scalar_metrics(
    model_name:str,
    model: Any,
    Xtest: AD,
    ytest: AS,
    ypreds: AS,
    desc: str='',
    df_eval:DN =None,
    style_col: str="Recall",
    show: bool=True,
    round_: IN=None
    )-> DN:
    """Get some scalar metrics for binary classification.

    Parameters
    -----------
    model_name: string
        Name of the model
    model: object
        sklearn model instance
    Xtest: np.array
        Test predictor variable
    ytest: np.array
        Test response variable
    ypreds: np.array
        Test response predictions.
    df_eval: pandas.DataFrame
        Pandas dataframe for model evaluation.
    style_col: string
        Name of the evaluation output column to style.
    show: bool
        Whether to show or not the styled dataframe
    round: int
        Rounding number to round off the displayed data.

.. code-block:: python

    '''

    This is a helper function intended to be used in Jupyter notebooks.

    This function displays various scalar binary classification metrics.
    AA===================================================================

    Accuracy = (TP+TN) / (TP+TN+FP+FN)   Overall accuracy of the model.
    Precision = TP / (TP+FP)  Precision has all P's.
    Useful for email spam detection.
    Recall = TP / (TP+FN) FP is replaced by FN.
    Useful for fraud detection, patient detection.
    F1 = 2 / (1/precision + 1/recall) Harmonic mean of precision and recall.

    F_beta = (1+beta^2) P R / (beta^2 P + R)
    F1 = TP / (TP + 0.5FP + 0.5FN)
    F2 = TP / (TP + 0.2FP + 0.8FN)
    F3 = TP / (TP + 0.1FP + 0.9FN)
    Useful when both precision and recall are important.
    AA===================================================================

    Matthews Correlation Coefficient
    https://www.wikiwand.com/en/Matthews_correlation_coefficient
    NOTE: F1-score depends on which class is defined as the positive class.
    F1-score will be high if majority of the classes defined are positive
    or vice versa.

    Matthews Correlation Coefficient is independent of classes frequencies.
    It is given by:

    MCC =     (TP * TN - FP*FN)
        ------------------------------------------------
        sqrt( (TP+FP)   (TP+FN)   (TN+FP)   (TN+FN)  )
    AA===================================================================

    Cohen's Kappa
    https://psychology.wikia.org/wiki/Cohen%27s_kappa
    Cohen's kappa measures the agreement between two raters who each classify
    N items into C mutually exclusive categories.

    Cohen's Kappa is given by:
    k = Pr(a) - Pr(e)
        --------------------
        1 - Pr(e)

    where Pr(a) is the relative observed agreement among raters,
    and Pr(e) is the hypothetical probability of chance agreement,
    using the observed data to calculate the probabilities
    of each observer randomly
    saying each category.
    If the raters are in complete agreement then κ = 1.
    If there is no agreement among the raters
    (other than what would be expected by chance)
    then κ ≤ 0.
    AA===================================================================


    By default the output result is sorted based on Recall.
    Usually Recall is the metric we are interested such as
    in case of classification Fraud Detection Modellings.

    In case of spam email detection, we are interested
    in "Precision" of the Spam detection and
    we can choose style_col as 'Precision'.

    In the classification cases such as cat and dogs classification,
    we may be equally interested in both precision and recall and
    we can choose 'F1' as the style column.

    Example:
    ------------
    df_eval = get_binary_classification_scalar_metrics1(
        model_name_lr,
        clf_lr,
        Xtest,ytest,
        ypreds_lr,
        desc="", df_eval=None) # or, df_eval = df_eval
    '''

    """

    # imports
    from sklearn.metrics import (accuracy_score,precision_score,
                                recall_score,f1_score,fbeta_score,
                                matthews_corrcoef,
                                average_precision_score,roc_auc_score,
                                cohen_kappa_score)
    if  not isinstance(df_eval, pd.DataFrame):
        df_eval = pd.DataFrame({'Model': [],
                                'Description':[],
                                'Accuracy':[],
                                'Precision':[],
                                'Recall':[],
                                'F0.5':[],
                                'F1':[],
                                'F2':[],
                                'Mathews_Correlation_Coefficient': [],
                                'Cohens_Kappa': [],
                                'Area_Under_Precision_Recall_Curve': [],
                                'Area_Under_ROC_Curve': [],
                            })
    # there is name collision for precision
    p = '{:.' + str(round_) + 'f}'

    # scalar metrics
    acc = accuracy_score(ytest,ypreds)
    precision = precision_score(ytest,ypreds,zero_division=0)
    recall = recall_score(ytest,ypreds,zero_division=0)
    f05 = fbeta_score(ytest,ypreds,beta=0.5,zero_division=0)
    f1 = f1_score(ytest,ypreds,zero_division=0)
    f2 = fbeta_score(ytest,ypreds,beta=2,zero_division=0)
    mcc = matthews_corrcoef(ytest,ypreds)
    kappa = cohen_kappa_score(ytest, ypreds)

    # area under the curves
    try:
        y_score = model.predict_proba(Xtest)[:,1]
    except:
        y_score = model.decision_function(Xtest)

    auprc = average_precision_score(ytest, y_score)
    auroc = roc_auc_score(ytest, y_score)

    row = [model_name,desc,acc,precision,recall,f05,f1,f2,mcc,kappa,auprc,auroc]

    df_eval.loc[len(df_eval)] =  row
    df_eval = df_eval.drop_duplicates()\
                .sort_values(style_col,ascending=False)
    df_eval.index = range(len(df_eval))

    df_style = (df_eval.style.apply(lambda ser:
                ['background: lightblue'
                if ser.name ==  style_col
                else '' for _ in ser])
                )
    caption = model_name + ' (' + desc + ')' if desc else model_name
    df_style.set_caption(caption)

    # rounding
    if round_:
        df_style = df_style.format({'Accuracy': p })
        df_style = df_style.format({'Precision': p })
        df_style = df_style.format({'Recall': p })
        df_style = df_style.format({'F0.5': p })
        df_style = df_style.format({'F1': p })
        df_style = df_style.format({'F2': p })
        df_style = df_style.format({'Mathews_Correlation_Coefficient': p })
        df_style = df_style.format({'Cohens_Kappa': p })

    if show:
        display(df_style)

    return df_eval

def get_binary_classification_scalar_metrics2(model_name:str,
    ytest:AS,
    ypreds:AS,
    desc:str='',
    df_eval:DN=None,
    style_col:str="F1",
    show:bool=True,
    round_:IN=None
    )-> DN:
    """Get some scalar metrics for binary classification.

    Parameters
    -----------
    model_name: string
        Name of the model
    ytest: np.array
        Test response variable
    ypreds: np.array
        Test response predictions.
    desc: string
        Description of the model. (string sentence)
    df_eval: pandas.DataFrame
        Pandas dataframe for model evaluation.
    style_col: string
        Name of the evaluation output column to style.
    show: bool
        Whether to show or not the styled dataframe
    round: int
        Rounding number to round off the displayed data.

.. code-block:: python

    '''

    This is a helper function intended to be used in Jupyter notebooks.

    This function displays various scalar binary classification metrics.

    AA===================================================================

    Accuracy = (TP+TN) / (TP+TN+FP+FN)   Overall accuracy of the model.
    Precision = TP / (TP+FP)  Precision has all P's.
    Useful for email spam detection.
    Recall = TP / (TP+FN) FP is replaced by FN.
    Useful for fraud detection, patient detection.
    F1 = 2 / (1/precision + 1/recall) Harmonic mean of precision and recall.

    F_beta = (1+beta^2) P R / (beta^2 P + R)
    F1 = TP / (TP + 0.5FP + 0.5FN)
    F2 = TP / (TP + 0.2FP + 0.8FN)
    F3 = TP / (TP + 0.1FP + 0.9FN)
    Useful when both precision and recall are important.

    AA===================================================================
    Matthews Correlation Coefficient
    https://www.wikiwand.com/en/Matthews_correlation_coefficient
    NOTE: F1-score depends on which class is defined as the positive class.
    F1-score will be high if majority of the classes defined are positive
    or vice versa.

    Matthews Correlation Coefficient is independent of classes frequencies.
    It is given by:

    MCC =     (TP * TN - FP*FN)
        ------------------------------------------------
        sqrt( (TP+FP)   (TP+FN)   (TN+FP)   (TN+FN)  )

    AA===================================================================

    Cohen's Kappa
    https://psychology.wikia.org/wiki/Cohen%27s_kappa
    Cohen's kappa measures the agreement between two raters who each classify
    N items into C mutually exclusive categories.

    Cohen's Kappa is given by:
    k = Pr(a) - Pr(e)
        --------------
            1 - Pr(e)

    where Pr(a) is the relative observed agreement among raters,
    and Pr(e) is the hypothetical probability of chance agreement,
    using the observed data to calculate the probabilities
    of each observer randomly
    saying each category.
    If the raters are in complete agreement then κ = 1.
    If there is no agreement among the raters
    (other than what would be expected by chance)
    then κ ≤ 0.

    AA===================================================================

    By default the output result is sorted based on F1.
    Usually Recall is the metric we are interested such as
    in case of classification Fraud Detection Modellings.

    In case of spam email detection, we are interested
    in "Precision" of the Spam detection and
    we can choose style_col as 'Precision'.

    In the classification cases such as cat and dogs classification,
    we may be equally interested in both precision and recall and
    we can choose 'F1' as the style column.
    '''

    Example
    ------------
    .. code-block:: python

        df_eval = get_binary_classification_scalar_metrics2(
                    'Statsmodels Logistic Regression',
                    y,ypreds,desc='',df_eval=None,style_col="F1",show=True)

    """

    # imports
    from sklearn.metrics import (accuracy_score,precision_score,
                                recall_score,f1_score,fbeta_score,
                                matthews_corrcoef,
                                average_precision_score,roc_auc_score,
                                cohen_kappa_score)
    if  not isinstance(df_eval, pd.DataFrame):
        df_eval = pd.DataFrame({'Model': [],
                                'Description':[],
                                'Accuracy':[],
                                'Precision':[],
                                'Recall':[],
                                'F0.5':[],
                                'F1':[],
                                'F2':[],
                                'Mathews_Correlation_Coefficient': [],
                                'Cohens_Kappa': []
                            })

    # there is name collision for precision
    p = '{:.' + str(round_) + 'f}'

    # scalar metrics
    acc = accuracy_score(ytest,ypreds)
    precision = precision_score(ytest,ypreds,zero_division=0)
    recall = recall_score(ytest,ypreds,zero_division=0)
    f05 = fbeta_score(ytest,ypreds,beta=0.5,zero_division=0)
    f1 = f1_score(ytest,ypreds,zero_division=0)
    f2 = fbeta_score(ytest,ypreds,beta=2,zero_division=0)
    mcc = matthews_corrcoef(ytest,ypreds)
    kappa = cohen_kappa_score(ytest, ypreds)

    # row
    row = [model_name,desc,acc,precision,recall,f05,f1,f2,mcc,kappa]

    df_eval.loc[len(df_eval)] =  row
    df_eval = df_eval.drop_duplicates()\
                .sort_values(style_col,ascending=False)
    df_eval.index = range(len(df_eval))

    df_style = (df_eval.style.apply(lambda ser:
                ['background: lightblue'
                if ser.name ==  style_col
                else '' for _ in ser])
                )
    caption = model_name + ' (' + desc + ')' if desc else model_name
    df_style.set_caption(caption)

    # rounding
    if round_:
        df_style = df_style.format({'Accuracy': p })
        df_style = df_style.format({'Precision': p })
        df_style = df_style.format({'Recall': p })
        df_style = df_style.format({'F0.5': p })
        df_style = df_style.format({'F1': p })
        df_style = df_style.format({'F2': p })
        df_style = df_style.format({'Mathews_Correlation_Coefficient': p })
        df_style = df_style.format({'Cohens_Kappa': p })

    if show:
        display(df_style)

    return df_eval

def get_binary_classification_report(
    model_name:str,
    ytest:AS,
    ypreds:AS,
    desc:bool="",
    df_clf_report:DN=None,
    style_col:str='Recall_1',
    show:bool=True
    )->DN:
    """Get binary classification report.

    Parameters
    -----------
    model_name: string
        Name of the model
    ytest: np.array
        Test response variable
    ypreds: np.array
        Test response predictions.
    desc: string
        Description of the model. (string sentence)
    df_clf_report: pandas.DataFrame
        Pandas dataframe for model classification report.
    style_col: string
        Name of the evaluation output column to style.
    show: bool
        Whether to show or not the styled dataframe

.. code-block:: python

    '''

    This is a helper function intended to be used in Jupyter notebooks.

    This function displays binary classification report
    (precision, recall, and f1-score)
    for all the classes and it also gives the number of supports
    (frequency of classes in ytest).

    By default the output result is sorted based on Recall_1.
    Usually Recall_1 is the class related to Frauds == 1
    and and we are interested in the metric
    "Recall" of the Fraud for Fraud Detection Modellings.

    In case of spam email detection, we are interested in "Precision"
    of the Spam detection and we can choose style_col as 'Precision_1'.

    In the classification cases such as cat and dogs classification,
    we may be equally interested in both precision and recall and
    we can choose 'F1_Score_1' as the style column.
    '''

    Examples
    ------------
    .. code-block:: python

        df_clf_report = get_binary_classification_report1("Logistic Regression",
        ytest,
        ypreds_lr,
        desc='',
        style_col='recall_1',
        df_clf_report=None)

    """
    if  not isinstance(df_clf_report, pd.DataFrame):
        df_clf_report = pd.DataFrame({'Model':[],
                                'Description':[],
                                'Precision_0': [],
                                'Precision_1':[],
                                'Recall_0':[],
                                'Recall_1':[],
                                'F1_Score_0':[],
                                'F1_Score_1':[],
                                'Support_0':[],
                                'Support_1':[]})

    from sklearn.metrics import precision_recall_fscore_support

    row = [model_name,desc] + np.array(precision_recall_fscore_support(
                                ytest,ypreds)).ravel().tolist()

    df_clf_report.loc[len(df_clf_report)] = row
    df_clf_report = df_clf_report.drop_duplicates()\
                    .sort_values(style_col,ascending=False)
    df_clf_report.index = range(len(df_clf_report))
    df_style = (df_clf_report.style
                    .apply(lambda ser:
                            ['background: lightblue' if ser.name ==  style_col
                            else '' for _ in ser])
                    )
    caption = model_name + ' (' + desc + ')' if desc else model_name
    df_style.set_caption(caption)
    if show:
        display(df_style)

    return df_clf_report

def print_confusion_matrix(
    model_name:str,
    ytest:AS,
    ypreds:AS,
    zero:str,
    one:str,
    color_zero:str='lightblue',
    color_one:str='salmon',
    show:bool=True
    )-> DN:
    """Print confusion matrix for binary classification.

    Parameters
    -----------
    model_name: string
        Name of the model.
    ytest: np.array
        Test response variable.
    ypreds: np.array
        Test response variable predictions.
    zero: str
        Zero label eg. No_Fraud
    one: str
        One label eg. Fraud
    color_zero: str
        Color for zero label eg. lightblue
    color_one: str
        Color for one label eg. salmon
    show: bool
        Show or not the styled dataframe.

    Example
    --------
    .. code-block:: python

        print_confusion_matrix('Logistic Regression', ytest,ypreds_lr,'Normal','Fraud')

    """
    from sklearn.metrics import confusion_matrix

    ytest = np.array(ytest).ravel()
    ypreds = np.array(ypreds).ravel()

    cm = confusion_matrix(ytest,ypreds)

    total_zeros = len(ytest[ytest==0])
    correct_zeros = cm[0][0]
    incorrect_zeros = cm[0][1]
    zeros_detection = correct_zeros / total_zeros * 100

    total_ones = len(ytest[ytest==1])
    correct_ones = cm[1][1]
    incorrect_ones = cm[1][0]
    ones_detection = correct_ones / total_ones * 100

    columns = [f"Predicted_{zero}", f"Predicted_{one}"]
    index = [zero, one]

    df_confusion = pd.DataFrame(cm,columns=columns,index = index)

    df_confusion[f'Total_{zero}'] = total_zeros
    df_confusion[f'Correct_{zero}'] = correct_zeros
    df_confusion[f'Incorrect_{zero}'] = incorrect_zeros
    df_confusion[f'{zero}_Detection'] = zeros_detection

    df_confusion[f'Total_{one}'] = total_ones
    df_confusion[f'Correct_{one}'] = correct_ones
    df_confusion[f'Incorrect_{one}'] = incorrect_ones
    df_confusion[f'{one}_Detection'] = ones_detection

    df_style = df_confusion.style.background_gradient(subset=[f'Predicted_{zero}'])
    df_style = df_style.background_gradient(subset=[f'Predicted_{one}'])
    df_style = df_style.set_caption(model_name)
    df_style = df_style.format("{:,.0f}")
    df_style = df_style.format({f"{zero}_Detection": lambda x: "{:.2f}%".format(x)})
    df_style = df_style.format({f"{one}_Detection": lambda x: "{:.2f}%".format(x)})
    df_style = df_style.apply(lambda x: ['background: '+ color_zero
                            if x.name == f'{zero}_Detection' else '' for _ in x])
    df_style = df_style.apply(lambda x: ['background: '+ color_one
                            if x.name == f'{one}_Detection' else '' for _ in x])

    if show:
        display(df_style)

    return df_confusion

def get_false_negative_frauds(
    model_name:str,
    ytest:ARR,
    ypreds:ARR,
    desc:str="",
    df_false_negatives:DN=None,
    show:bool=True
    )-> DN:
    """Get False Negative Frauds in a binary classification.

    Parameters
    -----------
    model_name: string
        Name of the model.
    ytest: np.array
        Test response variable.
    ypreds: np.array
        Test response variable predictions.
    desc: string
        A sentence describing the model.
    df_false_negatives: pd.DataFrame
        Pandas dataframe of False Negatives.
    show: bool
        Whether to show or styled the dataframe or not.

    Example
    --------
    .. code-block:: python

        '''

        This is a helper function intended to be used in Jupyter notebooks.

        This function displays a pandas dataframe having counts of total
        frauds and false negative counts of detecting the frauds.
        '''

    Example
    ---------

    .. code-block:: python

        df_false_negatives = get_false_negative_frauds('Logistic Regression',
        ytest,ypreds_best_lr,
        desc="Undersample, grid search",
        df_false_negatives=None,
        show=True);

    """
    style_col = 'Incorrect_Frauds'
    if  not isinstance(df_false_negatives, pd.DataFrame):
        df_false_negatives = pd.DataFrame({'Model':[],
                                'Description':[],
                                'Total_Frauds': [],
                                'Incorrect_Frauds':[],
                                'Incorrect_Percent':[],
                                })

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(ytest,ypreds)
    total_frauds = len(ytest[ytest==1])
    incorrect_frauds = cm[1][0]
    incorrect_percent = incorrect_frauds / total_frauds * 100

    row = [model_name,desc,total_frauds, incorrect_frauds,incorrect_percent]
    df_false_negatives.loc[len(df_false_negatives)] = row
    df_false_negatives = df_false_negatives.drop_duplicates()\
                            .sort_values(style_col,ascending=True)

    df_false_negatives.index = range(len(df_false_negatives))
    df_style = (df_false_negatives
                    .style
                    .apply(lambda ser:
                            ['background: lightblue' if ser.name ==  style_col
                            else '' for _ in ser])
                    )
    df_style = df_style.format({"Incorrect_Percent": lambda x: "{:.2f} %".format(x)})
    if show:
        display(df_style)

    return df_false_negatives

def plot_confusion_matrix_plotly(
    ytest:AS,
    ypreds:AS,
    labels:ARR,
    precision:int=4
    ):

    """Plot confusion using plotly.
    Parameters
    -----------
    ytest: np.array
        Test response variable.
    ypreds: np.array
        Test response variable predictions.
    labels: list
        A list containing labels of binary classification labels.
        For example `['Not-Fraud', 'Fraud']`.
    precision: int
        Integer to represent the precision.

    """

    import  plotly.figure_factory  as  ff
    from sklearn.metrics import confusion_matrix
    from plotly.offline import plot, iplot, init_notebook_mode

    xlabel = ['Predicted_' + i for i in labels] + ['Total'] + ['Accuracy %']
    ylabel = (labels + ['Total'])[::-1] # we need to reverse for plotly

    z = confusion_matrix(ytest,ypreds)
    z = np.c_[z,z.sum(1)]
    z = np.vstack([ z.sum(0), z])
    z = z[::-1]
    z = np.roll(z,1,axis=0)
    z = np.c_[z,[1.0,1.0,1.0]]
    z[2,3] = z[2,0] /  z[2,2] * 100 # category 1 accuracy
    z[1,3] = z[1,1] /  z[1,2] * 100 # category 2 accuracy
    z[0,3] = (z[2,0] + z[1,1]) / len(ytest) * 100 # total accuracy
    z = z.round(precision)

    fig  =  ff.create_annotated_heatmap(z, x=xlabel, y=ylabel,
                                        annotation_text=z.astype(str))
    init_notebook_mode(connected=True)
    iplot(fig)

def plot_roc_auc(
    ytest:AS,
    yprobs:AS
    ):
    """Plot Area Under the Curve.

    Parameters
    -----------
    ytest: list like
        actual test labels (0s or 1s)
    yprobs: list like
        Predicted probabilites for each instance.

    Example:
    ---------
    yprobs = model.predict_proba(Xtest)
    yprobs = yprobs[:][:,1] # choose only probs for 1
    plot_roc_auc(ytest,yprobs)

    """
    from sklearn import metrics

    yprobs = np.array(yprobs)

    # make 1d array
    if yprobs.shape[1] == 2:
        yprobs = yprobs[:][:,1] # choose only probs for 1

    # fpr and tpr
    fpr, tpr, thresholds = metrics.roc_curve(ytest,  yprobs)
    auc = metrics.roc_auc_score(ytest, yprobs)

    plt.plot(fpr,tpr,label=f"AUC={auc:.4f}")
    plt.plot(fpr,fpr,ls='--',color='blue',label='Random Guess')
    plt.legend(loc=4)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show()

def plot_roc_skf(
    clf:Any,
    X:AD,
    y:AS,
    skf:IN=None,
    random_state:IN=None,
    ofile:str=''
    ):
    """Plot the ROC Curve for the whole data X,y using stratified k-fold.

    Parameters
    -----------
    clf: object
        Classfier object from sklearn.
    X: np.ndarray
        Predictor variable data.
    y: np.ndarray
        Response variable.
    skf: int
        Stratified K-folding.
    random_state: int
        Random seed.
    ofile: string
        Name of output image file.

.. code-block:: python

        '''

        For k-fold, we do not need Xtrain and Xtest, we get them from X and y.

        This plot shows ROC curve for different folds of the k-fold.

        Ref: https://scikit-learn.org/stable/auto_examples/model_selection/
            plot_roc_crossval.html#sphx-glr-auto-examples
            -model-selection-plot-roc-crossval-py

        NOTE:
        ROC curves typically feature true positive rate on the Y axis,
        and false positive rate on the X axis.

        This means that the top left corner of the plot is the
        “ideal” point - a false positive rate of zero,
        and a true positive rate of one.

        This is not very realistic, but it does mean that
        a larger area under the curve (AUC) is usually better.

        '''

    """
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    from sklearn.model_selection import StratifiedKFold

    # make sure X and y are numpy arrays
    if isinstance(X,pd.core.frame.DataFrame):
        X = X.to_numpy()

    if isinstance(y,pd.core.frame.DataFrame):
        y = y.to_numpy()

    if isinstance(y,pd.core.frame.Series):
        y = y.to_numpy()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    cv = skf if skf else StratifiedKFold(n_splits=5,
                                        shuffle=True,
                                        random_state=random_state)
    for idx_tr, idx_tx in cv.split(X, y):

        # probabilites
        probas_ = clf.fit(X[idx_tr], y[idx_tr]).predict_proba(X[idx_tx])

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[idx_tx], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Minimum AUC ROC', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    title = 'Receiver Operating Characteristic Curves\n'
    title += '            for 5 fold Cross Validation'
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title,fontsize=15)
    plt.legend(loc="lower right")

    if ofile:
        plt.savefig(ofile,dpi=300)
    plt.show()