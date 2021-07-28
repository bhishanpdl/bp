#!/usr/bin/env sh

######################################################################
# @author      : Bhishan (Bhishan@BpMacpro.local)
# @file        : get_html
# @created     : Friday Oct 23, 2020 22:58:29 EDT
#
# @description : Get html files from jupyter notebooks 
######################################################################
rm -rf html/*.html && jupyter nbconvert *.ipynb --to html && mv *.html html


