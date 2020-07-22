# Install new kernel called nlp
```
# Create new env 
conda create -n nlp python=3.7
source activate nlp

# adding new kernel to ipython
conda install ipykernel  
python -m ipykernel install --user --name nlp --display-name "Python37 (nlp)"

# FIRST install fuzzy using pip (it failed for dataSc environment when installed later)
which pip # /Users/poudel/miniconda3/envs/nlp/bin/pip
/Users/poudel/miniconda3/envs/nlp/bin/pip install fuzzy

# deep learning
conda install -n nlp -c conda-forge tensorflow keras

# ipython
/Users/poudel/miniconda3/envs/nlp/bin/pip install ipywidgets

# linting
conda install -n nlp -c conda-forge autopep8  yapf black # needed for jupyter linting

# data manipulation
conda install -n nlp -c conda-forge numpy scipy pandas dask
conda install -n nlp -c conda-forge xlrd pytables pyarrow
/Users/poudel/miniconda3/envs/nlp/bin/pip install pandasql

# plotting
conda install -n nlp -c conda-forge seaborn
conda install -n nlp -c conda-forge plotly # I have not used these ==> bqplot bokeh holoviews
/Users/poudel/miniconda3/envs/nlp/bin/pip install plotly_express # express needed pip

# garbage collection
conda install -n nlp -c conda-forge gc

# file reader
/Users/poudel/miniconda3/envs/nlp/bin/pip install codecs

# machine learning
conda install -n nlp -c conda-forge scikit-learn scikit-image scikit-optimize
conda install -n nlp -c conda-forge imbalanced-learn

# print
/Users/poudel/miniconda3/envs/nlp/bin/pip install pprint

# nlp
conda install -n nlp -c conda-forge nltk textblob wordcloud autocorrect spacy gensim tweepy docx PyPDF2