#!/usr/bin/env sh

######################################################################
# @author      : Bhishan (Bhishan@BpMacpro.local)
# @file        : sync_github
# @created     : Tuesday Jul 07, 2020 18:47:32 EDT
#
# @description : Sync module with github 
######################################################################
rsync -azvu . ~/github/bp/examples/
cd ~/github/bp
git pull
git add --all
git commit -m "updated notebooks"
git push
cd -

