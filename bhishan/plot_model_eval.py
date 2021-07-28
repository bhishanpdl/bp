__author__ = 'Bhishan Poudel'

__doc__ = """
This module provides various model evaluation helper functions.

- plotly_binary_clf_evaluation(model_name,model,
    ytest,ypreds,y_score,df_train)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.plotly_binary_clf_evaluation(model_name,model,
    ytest,ypreds,y_score,df_train)

"""
__all__ = ["plotly_binary_clf_evaluation"]

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
import matplotlib
import matplotlib.pyplot as plt
import os
import time

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve,precision_recall_curve

def plotly_binary_clf_evaluation(model_name,model,
    ytest,ypreds,y_score,df_train,
    ofile=None,show=True,auto_open=False) :
    """Plot the binary classification model evaluation.

    Parameters
    -----------
    model_name: str
        Name of the model.
    model: object
        Fitted model. eg. RandomForestClassifier
    ytest: np.array
        Test array.
    ypreds: np.array
        Prediction array
    y_score: np.array
        Probability array.
    df_train: pd.DataFrame
        Input training dataframe.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    Examples
    ---------
    .. code-block:: python
        bp.plotly_binary_clf_evaluation(model_name,model,
    ytest,ypreds,y_score,df_train)

    """
    import plotly
    from sklearn.metrics import confusion_matrix
    import plotly.offline as py
    py.init_notebook_mode(connected=True)
    import plotly.graph_objs as go
    import plotly.tools as tls
    import plotly.figure_factory as ff

    #Conf matrix
    conf_matrix = confusion_matrix(ytest, ypreds)
    trace1 = go.Heatmap(z = conf_matrix  ,x = ["0 (Predicted)","1 (Predicted)"],
                        y = ["0 (True)","1 (True)"],xgap = 2, ygap = 2,
                        colorscale = 'Viridis', showscale  = False)

    #Show metrics
    tp = conf_matrix[1,1]
    fn = conf_matrix[1,0]
    fp = conf_matrix[0,1]
    tn = conf_matrix[0,0]
    Accuracy  =  ((tp+tn)/(tp+tn+fp+fn))
    Precision =  (tp/(tp+fp))
    Recall    =  (tp/(tp+fn))
    F1_score  =  (2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))))

    show_metrics = pd.DataFrame(data=[[Accuracy , Precision, Recall, F1_score]],
                                columns=['Accuracy', 'Precision', 'Recall', 'F1_score'])
    show_metrics = show_metrics.T.sort_values(by=0,ascending = False)

    y = show_metrics.index.values.tolist()

    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
    trace2 = go.Bar(x = (show_metrics[0].values),
                y = y,
                text = np.round_(show_metrics[0].values,4),
                textposition = 'auto',
                orientation = 'h', opacity = 0.8,marker=dict(
            color=colors,
            line=dict(color='#000000',width=1.5)))

    #Roc curve
    model_roc_auc = round(roc_auc_score(ytest, y_score) , 3)
    fpr, tpr, t = roc_curve(ytest, y_score)
    trace3 = go.Scatter(x = fpr,y = tpr,
                        name = "Roc : " + str(model_roc_auc),
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2),
                        fill='tozeroy')
    trace4 = go.Scatter(x = [0,1],y = [0,1],
                        line = dict(color = ('black'),width = 1.5,
                        dash = 'dot'))

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(ytest, y_score)
    trace5 = go.Scatter(x = recall, y = precision,
                        name = "Precision" + str(precision),
                        line = dict(color = ('lightcoral'),width = 2),
                        fill='tozeroy')

    #Cumulative gain
    pos = pd.get_dummies(ytest).values
    pos = pos[:,1]
    npos = np.sum(pos)
    index = np.argsort(y_score)
    index = index[::-1]
    sort_pos = pos[index]
    #cumulative sum
    cpos = np.cumsum(sort_pos)
    #recall
    recall = cpos/npos
    #size obs test
    n = ytest.shape[0]
    size = np.arange(start=1,stop=369,step=1)
    #proportion
    size = size / n
    #plots
    trace6 = go.Scatter(x = size,y = recall,
                        name = "Lift curve",
                        line = dict(color = ('gold'),width = 2), fill='tozeroy')

    #Feature importance
    if hasattr(model, 'feature_importances_'):
        coefficients  = pd.DataFrame(model.feature_importances_)
        column_data   = pd.DataFrame(list(df_train))
        coef_sumry    = (pd.merge(coefficients,column_data,left_index= True,
                                right_index= True, how = "left"))
        coef_sumry.columns = ["coefficients","features"]
        coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
        coef_sumry = coef_sumry[coef_sumry["coefficients"] !=0]
        trace7 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
                        name = "coefficients",
                        marker = dict(color = coef_sumry["coefficients"],
                                    colorscale = "Viridis",
                                    line = dict(width = .6,color = "black")))
        subplot_titles = ('Confusion Matrix',
                            'Metrics',
                            'ROC Curve'+" "+ '('+ str(model_roc_auc)+')',
                            'Precision - Recall Curve',
                            'Cumulative Gains Curve',
                            'Feature Importance',
                            )
        rows = 4
        specs = [[{}, {}],
                [{}, {}],
                [{'colspan': 2}, None],
                [{'colspan': 2}, None]]

    else:
        subplot_titles = ('Confusion Matrix',
                        'Metrics',
                        'ROC Curve'+" "+ '('+ str(model_roc_auc)+')',
                        'Precision - Recall Curve',
                        'Cumulative Gains Curve'
                        )
        rows = 3
        specs = [[{}, {}],
                [{}, {}],
                [{'colspan': 2}, None]
                ]

    #Subplots
    try:
        fig = plotly.subplots.make_subplots(rows=rows, cols=2, print_grid=False,
                        specs=specs,
                        subplot_titles=subplot_titles)
    except:
        fig = plotly.tools.make_subplots(rows=rows, cols=2, print_grid=False,
                        specs=specs,
                        subplot_titles=subplot_titles)

    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace3,2,1)
    fig.append_trace(trace4,2,1)
    fig.append_trace(trace5,2,2)
    fig.append_trace(trace6,3,1)
    if hasattr(model, 'feature_importances_'):
        fig.append_trace(trace7,4,1)


    fig['layout'].update(showlegend = False, title = '<b>Model Performance Report</b><br>('+str(model_name)+')',
                        autosize = False, height = 1500,width = 830,
                        plot_bgcolor = 'rgba(240,240,240, 0.95)',
                        paper_bgcolor = 'rgba(240,240,240, 0.95)',
                        margin = dict(b = 195))
    fig["layout"]["xaxis2"].update((dict(range=[0, 1])))
    fig["layout"]["xaxis3"].update(dict(title = "False Positive Rate"))
    fig["layout"]["yaxis3"].update(dict(title = "True Positive Rate"))
    fig["layout"]["xaxis4"].update(dict(title = "Recall"), range = [0,1.05])
    fig["layout"]["yaxis4"].update(dict(title = "Precision"), range = [0,1.05])
    fig["layout"]["xaxis5"].update(dict(title = "Percentage Contacted"))
    fig["layout"]["yaxis5"].update(dict(title = "Percentage Positive Targeted"))
    fig.layout.titlefont.size = 14
    fig['layout']['title']['x'] = 0.5
    if ofile:
        py.plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return py.iplot(fig)
