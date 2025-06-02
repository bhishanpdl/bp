# bp
[![Build Status](https://travis-ci.org/bhishanpdl/bp.svg?branch=master)](https://travis-ci.org/bhishanpdl/bp)[![Documentation Status](https://readthedocs.org/projects/bp/badge/?version=latest)](https://bp.readthedocs.io/en/latest/?badge=latest)

`bp` is my personal library for day to day use scripts such as data cleaning, machine learning, data visualization and so on.

# Usage
This module contains various data science and machine learning methods, which can seen in GIF videos in 'examples/gif' page.

# List of methods in the module
The module "bp" expands pandas DataFrame API and adds various visualization and data analysis functionalities. For example, we can get a plot of a numeric column using the method "df.bp.plot_num("my_numeric_variable")".

|                     |                     |                     |                     |
|---------------------|---------------------|---------------------|---------------------|
| BPAccessor          | hlp                 | plot_daily_cat      | plotly_corr_heatmap |
| Plotly_Charts       | json                | plot_date_cat       | plotly_countplot    |
| RandomColor         | light_axis          | plot_ecdf           | plotly_country_plot |
| add_interactions    | lm_plot             | plot_gini           | plotly_distplot     |
| add_text_barplot    | lm_residual_corr_plot | plot_ks           | plotly_histogram    |
| adjustedR2          | lm_stats            | plot_multiple_jointplots_with_pearsonr | plotly_mapbox |
| hex_to_rgb          | magnify             | plot_num            | plotly_radar_plot   |
| discrete_cmap       | multiple_linear_regression | plot_num_cat    | plotly_scattergl_plot |
| display_calendar    | no_axis             | plot_num_cat2       | plotly_scattergl_plot_colorcol |
| plot_plot_binn      | optimize_memory     | plot_num_num        | plotly_scattergl_plot_subplots |
| find_corr           | parallelize_dataframe | plot_pareto       | plotly_usa_bubble_map |
| freq_count          | parse_json_col      | plot_roc_auc        | plotly_usa_map      |
| get_binary_classification_report | partial_corr | plot_roc_skf    | plotly_usa_map2     |
| get_binary_classification_scalar_metrics | plot_boxplot_cats_num | plot_simple_linear_regression | point_biserial_correlation |
| get_binary_classification_scalar_metrics2 | plot_cat | plot_statistics | print_calendar |
| get_column_descriptions | plot_cat_binn   | plot_stem         | print_confusion_matrix |
| get_distinct_colors | plot_cat_cat        | plot_two_clusters | print_df_eval       |
| get_false_negative_frauds | plot_cat_cat2 | plotly_agg_country_plot | print_statsmodels_summary |
| get_high_correlated_features_df | plot_cat_cat_pct | plotly_agg_usa_plot | random |
| get_mpl_style       | plot_cat_num        | plotly_binary_clf_evaluation | regression_residual_plots |
| get_outliers        | plot_confusion_matrix_plotly | plotly_boxplot | remove_outliers |
| get_outliers_kde    | plot_corr           | plotly_boxplot_allpoints_with_outliers | rgb2hex |
| get_plotly_colorscale | plot_corr_style   | plotly_boxplot_categorical_column | select_kbest_features |
| get_yprobs_sorted_proportions | plot_corrplot_with_pearsonr | plotly_bubbleplot | show_methods |

# API Demo using GIF

### Stats Visualization
![Stats Example](examples/gif/stats.gif)

### Plots
![Plots Example](examples/gif/plots.gif)

### Timeseries Analysis
![Timeseries Example](examples/gif/timeseries.gif)

### Plotly Interactive Charts
![Plotly Example](examples/gif/plotly.gif)

### API Usage
![API Example](examples/gif/api.gif)

### Miscellaneous
![Misc Example](examples/gif/misc.gif)

# API References
For the usage of the module, visit API tutorials at ReadTheDocs or Github Pages.
- [API Usage of bp @ Read the Docs](https://bp.readthedocs.io/en/latest/)
- [API Usage of bp @ GitHub Pages](https://bhishanpdl.github.io/bp/)

# Motivation
This repository exists for two reasons.

1. Build a code base for personal use so that I don't have to write the same code snippet twice.
2. Learn the best practices in software industry.

# Examples

**Live Example**  
We can run the `bp` Jupyter notebooks live over the web at [Binder](http://mybinder.org):

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/bhishanpdl/bhishan)

**Rendered Preview**  

|  Notebook | Rendered   | Description  |  Author |
|---|---|---|---|
| demo.ipynb  | [ipynb](https://github.com/bhishanpdl/bp/blob/master/docs/notebooks/demo.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/bp/blob/master/docs/notebooks/demo.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |


# Contributors

* [Bhishan Poudel](https://bhishanpdl.github.io/) (Ph.D Physics, Ohio University)

# Installation
Go to your conda environment where you want to install this module, and then install the module.

## Method 01: Install final version
1. First go the folder where is setup.py located.
```bash
ls # there must be setup.py file.
which pip # make sure you are in right environment
python setup.py install --user # This will build some eggs and build directories.
```
- Once we install the module, we can not change the source codes.
- This command will install all the modules given in `setup.py` file.

## Method 02: Install Developer Version
- If we want to change the source code and update the module, we can install developer version.

```bash
which pip # make sure you are in correct conda environment
pip install -e .  # -e means editable version and dot is the path
                  # where there is setup.py file located.

# this will create: bp.egg-info folder.
# make sure you ignore this folder in .gitignore
```

# Using Docker Container
```bash
# step 00: Go to the path where you want to work on
mkdir -p ~/temp; cd ~/temp
mkdir example_docker; cd example_docker

# step 01: download the Dockerfile from this repository
wget https://raw.githubusercontent.com/bhishanpdl/bp/master/Dockerfile
clear;ls
cat Dockerfile

# step 02: Run the docker in your mac (you will see docker icon on menubar)

# step 03: Build the docker image and give name "bp"
# this takes about 2 minutes
# this installs vim, jupyter and some latex modules needed for jupyter-notebook
# --no-cache will delete previous cache
# -t will allow accessing docker image using terminal
docker build --no-cache -t bp .

# step 04: Run the docker image
# allow terminal using -ti
# allow port using -p

docker run -it -p 8888:8888 bp

# if this gives error, try another port
docker run -it -p 8889:8889 bp

# If you have already running this container
# docker ps
# docker stop container_id_obtained_from_docker_ps
# this will open python, do not close it.
# open the new tab on the terminal to go inside the container.

# step 05: Go to docker terminal and run commands there
docker ps
docker exec -ti CONTAINER_ID_from_docker_ps /bin/sh

ls
pwd

# run python script
cd /home/bp/docs/scripts;clear;ls
which python
python --version

python example_json.py

# run jupyter notebook
cd /home/bp/docs/notebooks
# open the second link in the browser, where we can run the notebook.
# hit ctrl c to exit the python
# close the terminal tab where docker image was running.

# step 06: stop and remove the container

# first create new tab on the terminal
# hit ctrl d to close the running python

docker ps
docker stop CONTAINER_ID_from_docker_ps
docker rmi -f bp

Now close the docker app on your machine.
```

# License
This is my personal library intended to be used by only me.
It's not a public library.
