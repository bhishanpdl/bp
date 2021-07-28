# Outputs:
# docs/source  ==> _static, _templates and conf.py index.rst
# docs/build   ==> empty
/Users/poudel/opt/miniconda3/envs/tf2/bin/sphinx-quickstart -q -p "Bhishan" -a "Bhishan Poudel" -v 1 -r 1 \
    --ext-autodoc --ext-doctest --ext-viewcode --ext-imgmath \
    --no-batchfile --sep docs

# Now change the docs/source/conf.py
# uncomment: os sys and sys.path
sed -ie 's/# import os/import os/g' docs/source/conf.py
sed -ie 's/# import sys/import sys/g' docs/source/conf.py
sed -ie "s/# sys.path.insert(0, os.path.abspath('.'))/sys.path.insert(0, os.path.abspath('.'))/g" docs/source/conf.py

# Add path ../../
awk -v n=16 -v s="sys.path.insert(0, os.path.abspath('../../$1'))" 'NR == n {print s} {print}' \
    docs/source/conf.py > docs/source/conf_new.py;
    /bin/rm docs/source/conf.py; mv docs/source/conf_new.py docs/source/conf.py

# Add one more extension napolean to docs/source/conf.py
awk -v n=40 -v s="    'sphinx.ext.napoleon'," 'NR == n {print s} {print}' \
    docs/source/conf.py > docs/source/conf_new.py;
    /bin/rm docs/source/conf.py; mv docs/source/conf_new.py docs/source/conf.py

# Change the theme alabaster ==> classic in docs/source/conf.py
sed -ie "s/html_theme = 'alabaster'/html_theme = 'classic'/g" docs/source/conf.py


# Outputs:
# docs/build/html (which has index.html)
# docs/build/doctrees
cd docs; make html; cd -


# Automatically create rst files
# Example: sphinx-apidoc -o docs/source bp
# where `bhishan` folder has .py files.
#
# Outputs:
# docs/source/bhishan.rst
# docs/source/modules.rst
#
/Users/poudel/opt/miniconda3/envs/tf2/bin/sphinx-apidoc -o docs/source bhishan


# Replace some strings in docs/source/bhishan.rst
# bhishan.bp module ==> bhishan.bp
# But keep automodule:: ==> automodule::
#
for f in docs/source/*.rst; do sed -ie 's/module//g' $f; done
for f in docs/source/*.rst; do sed -ie 's/auto::/automodule::/g' $f; done

# delete .rste temp files
for f in docs/source/*.rste; do /bin/rm $f; done


# Delete source/index.rst and rename modules.rst to index.rst
mv docs/source/modules.rst docs/source/index.rst

# Add TOC to docs/source/index.rst
awk -v n=1 -v s=".. contents:: Table of Contents\n   :depth: 3\n\n" \
                    'NR == n {print s} {print}' \
                docs/source/index.rst > docs/source/tmp; mv docs/source/tmp docs/source/index.rst

# Rebuild docs/build/html/index.html
cd docs; /Users/poudel/opt/miniconda3/envs/tf2/bin/sphinx-build -b html source build/html; cd -

# Open the docs/build/html/index.html
open docs/build/html/index.html

# Command:
# rm -rf docs && bash mysphinx.sh bhishan  && open docs/build/html/index.html
