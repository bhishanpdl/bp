__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various statsmodels utilities.

- regression_residual_plots(model_fit, dependent_var,
    data, size = [10,10],cook_xlim=None,cook_ylim=None,
    annotate_outliers=True,verbose=True,title=None,
    loc='upper right',ofile=None)
- print_statsmodels_summary(summary,verbose=True)
- lm_stats(X, y, y_pred)
- lm_plot(df_X1train, df_ytrain, df_ypreds_train,verbose=True,ofile=None)
- lm_residual_corr_plot(df_X1train,df_ytrain,df_ypreds_train,
                            verbose=True,ofile=None)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.ml_statsmodels?

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import time

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.robust as smrb # smrb.mad() etc
from IPython.display import display, HTML


def regression_residual_plots(model_fit, dependent_var,
                                data, size = [10,10],
                                cook_xlim=None,
                                cook_ylim=None,
                                annotate_outliers=True,
                                verbose=True,
                                title=None,
                                loc='upper right',
                                ofile=None):
    """
    Parameters
    -----------
    model_fit: object
        Fitted model from statsmodels.
    dependent_var: string
        Feature name for dependent variable.
    data: pandas.DataFrame
    size: list
        default is [10,10]. matplotlib figsize = [10,10]

    .. code-block:: python

        import matplotlib.pyplot as plt
        import statsmodels.api as sm
        model_fit = sm.OLS(endog= DEPENDENT VARIABLE, exog= INDEPENDENT VARIABLE).fit()

    NOTE:
    -------
    Ive only run it on simple, non-robust, ordinary least squares models,
    but these metrics are standard for linear models.
    Ref: https://www.kaggle.com/nicapotato/in-depth-simple-linear-regression
    """

    # Extract relevant regression output for plotting
    # fitted values (need a constant term for intercept)
    model_fitted_y = model_fit.fittedvalues
    # model residuals
    model_residuals = model_fit.resid
    # normalized residuals
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    # leverage, from statsmodels internals
    model_leverage = model_fit.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model_fit.get_influence().cooks_distance[0]

    ########################################################################
    # Plot Size
    fig = plt.figure(figsize=size)

    # Residual vs. Fitted
    ax = fig.add_subplot(2, 2, 1) # Top Left
    sns.residplot(model_fitted_y, dependent_var, data=data,
                    lowess=True,
                    scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                    ax=ax)
    ax.set_title('Residuals vs Fitted')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')

    # Annotations of Outliers
    if annotate_outliers:
        abs_resid = model_abs_resid.sort_values(ascending=False)
        abs_resid_top_3 = abs_resid[:3]
        for i in abs_resid_top_3.index:
            ax.annotate(i, xy=(model_fitted_y[i], model_residuals[i]));

    ########################################################################
    # Normal Q-Q
    ax = fig.add_subplot(2, 2, 2) # Top Right
    QQ = sm.ProbPlot(model_norm_residuals)
    QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1, ax=ax)
    ax.set_title('Normal Q-Q')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Standardized Residuals')

    # Annotations of Outliers
    if annotate_outliers:
        abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            ax.annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                    model_norm_residuals[i]));

    ########################################################################
    # Scale-Location Plot
    ax = fig.add_subplot(2, 2, 3) # Bottom Left
    plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}, ax=ax)
    ax.set_title('Scale-Location')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('$\sqrt{|Standardized Residuals|}$');
    # Annotations of Outliers
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
    # This gives points at wrong places.
    # for i in abs_norm_resid_top_3:
    #     try:
    #         ax.annotate(i,
    #                     xy=(model_fitted_y[i],
    #                     model_norm_residuals_abs_sqrt[i]));
    #     except:
    #         pass


    ########################################################################
    # Cook's Distance Plot
    ax = fig.add_subplot(2, 2, 4) # Bottom Right
    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                ax=ax)
    ax.set_xlim(cook_xlim)
    ax.set_ylim(cook_ylim)
    ax.set_title('Residuals vs Leverage')
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')

    # Annotations
    if annotate_outliers:
        leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(i, xy=(model_leverage[i],model_norm_residuals[i]))

    # Shenanigans for Cook's distance contours
    def graph(formula, x_range, label=None,ls='--'):
        x = x_range
        y = formula(x)
        plt.plot(x, y, label=label, lw=1, ls=ls, color='red')
    p = len(model_fit.params) # number of model parameters
    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
            np.linspace(0.001, 0.200, 50),
            'Cook\'s distance 0.5') # 0.5 line
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
            np.linspace(0.001, 0.200, 50),label='Cook\'s distance 1.0',ls=':') # 1 line
    plt.legend(loc=loc)
    if title: plt.suptitle(title)
    if ofile:
        plt.savefig(ofile,bbox_inches='tight')

    html = """
<div class="alert alert-block alert-info">
<div align="center">
    <h3>Plot Diagnostics </h3>
</div>
<h5> <span style="color:green">1. Residuals vs Fitted Plot (Test of Linearity of Data) </span> </h5><br>
    - If <span style="color:red"> solid red line</span> is along y=0, assumption of linearity holds true.<br>
    - If we see U-shaped <span style="color:red"> solid red line</span>, then
        our linear model is not providing a optimal fit to our data.<br>
    - The relationship is non-linear and we may need to make quadratic
        (or, other polynomial transformation of the data.)<br>
    - If we see residuals more varied as we go right or left,
        the assumption of of constant variance of residuals is violated. (Data is
        heteroskedastic.)<br>
    - In case of heteroskedasticity, we may need to log1p, boxcox1p, sqrt or do other transformations
        to make the dataset more uniform.<br>

<h5> <span style="color:green">2.  Normal Q-Q plot (Test of Normality of Data)</span></h5><br>
    - If fitted points align with 45 degree line, assumption of normality holds true.<br>
    - Technical note: We need to use fit='True' parameter to compare fitted points
    with 45 degree.
    <code class="inlinecode"> smg.qqplot(model_fitted.resid_pearson,line='45',fit='True') </code>

<h5> <span style="color:green">3. Standardized Residuals Plot
    (Test of Constant Variance of Residuals, Homoskedasticity) </span></h5><br>
    - This is the plot after we standardize the residuals.<br>
    - If data points are clustered at left or right,
        the data exhibits heteroskedasticity.<br>
    - Assumption of constant variance of residual
        is violated.<br>
    - To reduce the heteroskedasticity, we may use non-linear transformations
        such as log1p, boxcox1p, sqrt, power(1/3) and so on.<br>
    - To reduce heteroskedasticity, we may use < a href="http://www.statsmodels.org/0.6.1/examples/notebooks/generated/wls.html" weighted least squares method. </a> <br>
    - If we see few outliers with high y-values, we have high variance residual outliers.<br>
    - If the dataset follows Normal distribution, we can use Tukey's IQR (Inter Quartile Range)
        method to detect and treat the outliers.<br>
    - If the dataset is non-normal, we can use non-parametric methods such as <a href="http://www.statsmodels.org/devel/generated/statsmodels.nonparametric.kde.KDEUnivariate.html">Kernel Density Estimation</a> to detetect and treat the outliers.<br>
    - Small residuals on y-axis is better.<br>

<h5> <span style="color:green">4. Leverage Plot for Residuals (Cook's Test of Outliers) </span></h5> <br>
    - Solid <span style="color:red"> red line</span> is leverage best fit line.<br>
    - Dashed <span style="color:red"> red line</span> is Cook's distance for 0.5 * p * (1-x) /x.<br>
    - Dotted <span style="color:red"> red line</span> is Cook's distance for 1 * p * (1-x) /x.<br>
    - p is number of model parameters. x is plot values: np.linspace(0.001, 0.200, 50).<br>
    - If the values exceeds the farthest <span style="color:red"> red line</span>,
        it has high leverage on the dataset and may be considered an outlier.<br>
    - If we data points beyond Studentized (or Standardized) Residual of value 3,
        the data point may be an outlier.

</div>
"""

    if verbose: display(HTML(html))
    plt.show()
    plt.close()

def print_statsmodels_summary(summary,verbose=True):
    """Print statsmodels fitted model summary with some color hightlights.

    Parameters
    -----------
    summary: object
        Statsmodel fitted model summary.

    Example
    --------
    .. code-block:: python

        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        formula = 'price ~ sqft_living'
        model = smf.ols(formula=formula, data= pd.concat([Xtrain,ytrain],axis=1))
        model_fit = model.fit()
        summary = model_fit.summary()

    Notes:
    ------
    Colors picked from: https://www.w3schools.com/colors/colors_picker.asp


    Example
    --------
    .. code-block:: python

        '''
        From: '<th>  Adj. R-squared:    </th>'
        To  : '<th style="background-color:{};"> Adj. R-squared: </th>'
                '<th                              Adj. R-squared:    </th>',
        Fill the color background inside <th> tag.
        '''
    """

    desc = """
<div class="alert alert-block alert-info">
<div align="center">
    <h3> Model Summary Description </h3>
</div>

<h5> <span style="color:green">1. R-squared (Coefficient of Determination) </span> </h5>

<a href="https://www.wikiwand.com/en/Coefficient_of_determination">Coefficient_of_determination</a><br>

    - R-squared = 1 - (SS_res / SS_tot)<br>
        SS_res = sum (y_i - f_i)**2  = sum(e_i **2)<br>
        SS_tot = sum (y_i - y_bar)**2<br>
    - If R2 = 0.49, then, 49% of the variability of the dependent
        variable has been accounted for, <br>
        and the remaining 51% of the
        variability is still unaccounted for.<br>

<h5> <span style="color:green">2. Adjusted R2 (R bar squared) </span></h5>
<a href="https://www.wikiwand.com/en/Coefficient_of_determination">Coefficient_of_determination</a><br>
    - R_bar_squared = 1 - (1-R2) * (n-1) / (n-p-1)<br>
    - The adjusted R2 can be negative.<br>
    - It will always be less than or equal to that of R2.<br>
    - Adjusted R2 can be interpreted as an unbiased<br>
        (or less biased) estimator of the population R2.<br>
    - R2 is a positively biased estimate of the population value.<br>
        When p increases, R2 increases, but R_bar_squared may not increase.<br>

<h5> <span style="color:green">3. F-statistic </span> </h5>
<a href="https://www.wikiwand.com/en/F-test">F-statistic</a><br>
    -  the one-way ANOVA F-test statistic is<br>
		<div class="alert alert-warning">
			&nbsp;&nbsp;&nbsp;&nbsp;explained variance<br>
		F=&nbsp;  --------------------<br>
		&nbsp;&nbsp;&nbsp;&nbsp;unexplained variance<br>

		</div>
    - When there are only two groups for the one-way ANOVA F-test,
        F = t**2, where t is the Student's t statistic.<br>
    - For two models 1 and 2,
				<div class="alert alert-warning">
				    &nbsp;&nbsp;&nbsp;&nbsp;(RSS1 - RSS2) / (p2-p1)<br>
				F=&nbsp; --------------------<br>
				&nbsp;&nbsp;&nbsp;&nbsp;(RSS2) / (n-p2)<br>
				</div>

<h5> <span style="color:green">4. AIC </span></h5>
<a href="https://www.wikiwand.com/en/Akaike_information_criterion">Akaike Information Criterion</a><br>
    - Akaike Information Criterion. <br>
    - AIC = 2k - 2 ln L  where k is the number of model parameters,
        L is log likelihood.<br>
    - Adjusts the log-likelihood based on the number of observations
        and the complexity of the model.<br>
    - Penalizes the model selection metrics when more independent
        variables are added.<br>

<h5> <span style="color:green">5. BIC </span> </h5>
<a href="https://www.wikiwand.com/en/Bayesian_information_criterion">Bayesian Information Criterion</a><br>
    - Bayesian Information Criterion.<br>
    - BIC = ln(n) * k - 2 ln L  where k is the number of model parameters,
        L is log likelihood.<br>
    - Similar to the AIC, but has a higher penalty for models with more parameters.<br>
    - Penalizes the model selection metrics when more independent variables are added.<br>
    -  BIC is only valid for sample size n much larger than
        the number k of parameters in the model.<br>

<h5> <span style="color:green">6. P > |t| </span></h5>
<a href="https://www.wikiwand.com/en/P-value">P-value</a><br>
    - p-value means probability value.<br>
    - p-value means that the null-hypothesis model parameter = 0 is true.<br>
    - If it is less than the confidence level, often 0.05,
        it indicates that there is a statistically significant
        relationship between the predictor and the response.<br>

<h5> <span style="color:green">7. Skewness  </span> </h5>
<a href="https://www.wikiwand.com/en/Skewness">Skewness</a><br>
    - A measure of the symmetry of the data about the mean.<br>
    - Normally-distributed errors should be symmetrically distributed about the mean.<br>
    - The normal distribution has 0 skew.<br>

<h5> <span style="color:green">8. Kurtosis </span></h5>
<a href="https://www.wikiwand.com/en/Kurtosis">Kurtosis</a><br>
    - A measure of the shape of the distribution.<br>
    - The normal distribution has a Kurtosis of 3.<br>
    - If kurtosis is greater than 3, curve is tall and peaked.<br>

<h5> <span style="color:green">9. Omnibus D Angostino’s test </span> </h5>
<a href="https://www.wikiwand.com/en/D%27Agostino%27s_K-squared_test">Omnibus D Angostino’s test</a><br>
    - It provides a combined statistical test for the presence of skewness and kurtosis.<br>
    - This is a goodness-of-fit measure of departure from normality,
        that is the test aims to establish whether or not the given sample
        comes from a normally distributed population.<br>
    - The test is based on transformations of the sample kurtosis and skewness,
        and has power only against the alternatives that the
        distribution is skewed and/or kurtic.<br>

<h5> <span style="color:green">10. Jarque-Bera </span></h5>
<a href="https://www.wikiwand.com/en/Jarque%E2%80%93Bera_test">Jarque-Bera Test</a><br>
    - A different test of the skewness and kurtosis.<br>
    - In statistics, the Jarque–Bera test is a goodness-of-fit test of
        whether sample data have the skewness and kurtosis
        matching a normal distribution.<br>
    - The test statistic is always nonnegative.<br>
    - If it is far from zero, it signals the data do not have a normal distribution.<br>

<h5> <span style="color:green">11. Durbin-Watson </span> </h5>
<a href="https://www.wikiwand.com/en/Durbin%E2%80%93Watson_statistic">Durbin-Watson Stastistic</a><br>
    - In statistics, the Durbin–Watson statistic is a test statistic
        used to detect the presence of autocorrelation at lag 1 in the residuals.<br>
    - Often important in time-series analysis.<br>
    - A similar assessment can be also carried out with the
        Breusch–Godfrey test and the Ljung–Box test.<br>

<h5> <span style="color:green">12. Cond. No </span></h5>
<a href="https://www.wikiwand.com/en/Condition_number">Condition Number</a><br>
    - The condition number of a function measures how much the output value
        of the function can change for a small change in the input argument.<br>
    - This is used to measure how sensitive a function is to changes
        or errors in the input.<br>
    - In linear regression the condition number of the moment matrix
        can be used as a diagnostic for multicollinearity.<br>
    - A problem with a low condition number is said to be well-conditioned.<br>
    - A problem with a high condition number is said to be ill-conditioned.<br>

</div>
    """
    if verbose: display(HTML(desc))

    summary_html = summary.as_html()

    to_color_html_tags = ['<th>  Adj. R-squared:    </th>',
                '<th>  F-statistic:       </th>',
                '<th>  AIC:               </th>',
                '<th>  BIC:               </th>',
                '<th>coef</th>',
                '<th>std err</th>',
                '<th>P>|t|</th>',
                '<th>[0.025</th>',
                '<th>0.975]</th>'
                ]

    light_background_colors = ['#aec7e8', '#c7e9c0','#bcbddc','#ffe699',
            '#808080','#ccccff','#ffd9b3','#ff9896',
            '#ff9896']

    # insert style background inside <th> tag.
    colored_html_tags  = ['<th style="background-color:{};"'.format(
        light_background_colors[i]) +
        to_color_html_tags[i].lstrip('<th')
        for i in range(len(to_color_html_tags)) ]

    for i in range(len(to_color_html_tags)):
        summary_html = summary_html.replace(to_color_html_tags[i],
                                            colored_html_tags[i])


    return HTML(summary_html)

################################
## ISLR Python
################################
# Functions to emulate R's lm().plot() functionality

def lm_stats(X, y, y_pred):
    """Get leverage and studentised residuals.

    Parameters
    -----------
    X: np.array
        Predictor variable.
    y: np.array
        Response variable.
    y_pred: np.array
        predictions of response variable.

    https://en.wikipedia.org/wiki/Studentized_residual#How_to_studentize

    """
    # Responses as np array vector
    try:
        y.shape[1] == 1
        # take first dimension as vector
        y = y.iloc[:,0]
    except:
        pass
    y = np.array(y)

    # Residuals
    residuals = np.array(y - y_pred)

    # Hat matrix
    H = np.array(X @ np.linalg.inv(X.T @ X)) @ X.T

    H = np.array(H)

    # Leverage
    h_ii = H.diagonal()

    ## Externally studentised residual
    # In this case external studentisation is most appropriate
    # because we are looking for outliers.

    # Estimate variance (externalised)
    σi_est = []
    for i in range(X.shape[0]):
        # exclude ith observation from estimation of variance
        external_residuals = np.delete(residuals, i)
        σi_est += [np.sqrt((1 / (X.shape[0] - X.shape[1] - 1))
                           * np.sum(np.square(external_residuals)))]
    σi_est = np.array(σi_est)

    # Externally studentised residuals
    t = residuals / σi_est * np.sqrt(1 - h_ii)


    # Return dataframe
    return pd.DataFrame( {'residual': residuals,
                        'leverage': h_ii,
                        'studentised_residual': t,
                        'y_pred': y_pred})


def lm_plot(df_X1train, df_ytrain, df_ypreds_train,verbose=True,ofile=None):
    """Provides R style residual plots based on results from lm_stats

    Parameters
    ----------
    df_X1train: pandas.DataFrame
        Pandas dataframe with bias column added.
    df_ytrain: pandas.Series
        Pandas series with response variable.
    df_ypreds_train: pandas.Series
        Pandas series of reponse variable predictions.
    verbose: bool
        Boolean to show whether to print description of stats or not.
    ofile: string
        Name of output file for plot.
    """
    from scipy import stats

    lm_stats_df = lm_stats(df_X1train, df_ytrain, df_ypreds_train)

    # Parse stats
    t      = lm_stats_df['studentised_residual']
    h_ii   = lm_stats_df['leverage']
    y_pred = lm_stats_df['y_pred']

    # setup axis for grid
    plt.figure(1, figsize=(16, 18))

    # Studentised residual plot
    plt.subplot(321)
    ax = sns.regplot(x=y_pred, y=t, lowess=True)
    plt.xlabel('Fitted Values')
    plt.ylabel('Studentised Residuals')
    plt.title('Externally Studentised Residual Plot', fontweight='bold')
    # Draw Hastie and Tibshirani's bounds for possible outliers
    ax.axhline(y=3, color='r', linestyle='dashed',label='y = ± 3')
    ax.axhline(y=-3, color='r', linestyle='dashed');
    ax.legend(loc='upper right')

    # Normal Q-Q plot
    plt.subplot(322)
    ax = stats.probplot(t, dist='norm', plot=plt)
    plt.ylabel('Studentised Residuals')
    plt.title('Normal Q-Q Plot', fontweight='bold')

    # Standardised residuals
    plt.subplot(323)
    ax = sns.regplot(x=y_pred, y=np.sqrt(np.abs(t)), lowess=True)
    plt.xlabel('Fitted Values')
    plt.ylabel('√Standardized Residuals')
    plt.title('Scale-Location', fontweight='bold')

    # Residuals vs Leverage plot
    plt.subplot(324)
    ax = sns.scatterplot(x=h_ii, y=t)
    plt.xlabel('Leverage')
    plt.ylabel('Studentised Residuals')
    plt.title('Externally Studentised Residual vs Leverage', fontweight='bold');
    if ofile:
        plt.savefig(ofile, bbox_inches='tight',dpi=300);
    desc = """
    1. Externally Studentised Residual Plot (Outliers Test):
    - The horizontal red dashed lines are studentised t values t = ± 3
    - The points outside t = ± 3 may be considered outliers.
    - If we see U-shaped fitted solid blue line, our data is non-linear.
        We might need to transoform features or include polynomial features.

    2. Normal Q-Q Plot (Test of Normality)
    - If fitted points align with 45 degree line,
        the assumption of normality is likey to hold true.

    3. Scale-Location Plot (Test of Constant Variance, homoskedasticity)
    - Small residuals on y-axis is better.
    - If we see conical shape, data is heteroskedastic.
    - If data points are clustered at left or right, we observe heteroskedasticity.
    - If we see few outliers with high y-values,
        we have high variance residual outliers.

    4. Residuals vs Leverage Plot (Outliers Test)
    - Studentised residual larger than 3 are potential outliers.
    - High leverage points are potential outliers.
    """

    if verbose: print(desc)
    plt.show()

def lm_residual_corr_plot(df_X1train,df_ytrain,df_ypreds_train,
                            verbose=True,ofile=None):
    """Linear models residual correlation plots.

    Parameters
    ----------
    df_X1train: pandas.DataFrame
        Pandas dataframe with bias column added.
    df_ytrain: pandas.Series
        Pandas series with response variable.
    df_ypreds_train: pandas.Series
        Pandas series of reponse variable predictions.
    verbose: bool
        Boolean to show whether to print description of stats or not.
    ofile: string
        Name of output file for plot.
    """
    lm_stats_df = lm_stats(df_X1train, df_ytrain, df_ypreds_train)
    r = lm_stats_df['residual']
    # Residuals correlation
    plt.figure(1, figsize=(16, 5))
    ax = sns.lineplot(x=list(range(r.shape[0])), y=r)
    plt.xlabel('Observation')
    plt.ylabel('Residual')
    plt.title('Correlation of Error Terms', fontweight='bold');
    desc = """
    1. Correlation of Error Terms (Collinearity Test):
    - If the magnitude of errors increase/decrease as we go along x-axis,
        the error terms may be correlated.
    -   This could mean that our estimated standard errors underestimate
        the true standard errors.
    - Our confidence and prediction intervals may be
        narrower than they should be.
    """

    if verbose: print(desc)
    if ofile: plt.savefig(ofile, bbox='tight',dpi=300)
    plt.show()