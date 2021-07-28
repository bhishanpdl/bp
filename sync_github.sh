#!/usr/bin/env sh

######################################################################
# @author      : Bhishan (Bhishan@BpMacpro.local)
# @file        : sync_github
# @created     : Tuesday Jul 07, 2020 18:47:32 EDT
#
# @description : Sync module with github
######################################################################

# sync to google colab
rsync -azvu bhishan ~/Google\ Drive/Colab\ Notebooks/Softwares/

# sync to github and git push
rsync -azvu . /Volumes/Media/github/bhishan
cd /Volumes/Media/github/bhishan
git pull
git add --all
git commit -m "updated module"
git push
cd -

