# Make documentation using Sphinx
```bash
# make sure you have sphinx installed in required conda env
which pip
pip freeze | grep -i sphinx
pip freeze | grep -i sphinx_rtd_theme

# if not installed, install them
/Users/poudel/miniconda3/envs/tf2/bin/pip install sphinx
/Users/poudel/miniconda3/envs/tf2/bin/pip install sphinx_rtd_theme

# Make sure you have sphinx-quickstart command available
find ~/miniconda3/envs/tf2 -name "sphinx-q*"

/Users/poudel/miniconda3/envs/tf2/bin/sphinx-quickstart

# Go to the docs folder, where is Makefile located.
make html # this will create html files in _build folder.

#make sure we do not have ignored _build in .gitignore file.

# check how the documentation is created
open _build/html/index.html

```
