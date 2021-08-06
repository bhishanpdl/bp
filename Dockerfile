FROM python:3.7

# Update the OS as needed
RUN apt-get update

# install vim so that you can use vim on docker terminal
RUN apt-get install -y nmap vim

# Get TEX fonts for jupyter notebook
RUN apt-get install texlive-latex-recommended -y
RUN apt-get install texlive-latex-extra -y
RUN apt-get install texlive-fonts-recommended -y
RUN apt-get install chktex -y
RUN apt-get install dvipng -y

# Install python requirements using pip:
RUN pip install --upgrade jupyter
RUN pip install nbconvert

# Install modules for development (not required in production)
#RUN pip install sphinx
#RUN pip install sphinx_rtd_theme

# Install the dependencies and the package:
WORKDIR /home
RUN git clone https://github.com/bhishanpdl/bp.git
WORKDIR /home/bp
RUN pip install -r requirements.txt
RUN python setup.py install

# cd to the work directory and launch a jupyter notebook on run
#WORKDIR /home/bp/docs/notebooks
#CMD jupyter notebook --no-browser --allow-root --port=8888 --ip=0.0.0.0
