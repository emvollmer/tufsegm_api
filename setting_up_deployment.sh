#!/bin/bash

# ########## Configure NextCloud RCLONE connection
if rclone listremotes | grep -q "rshare:" ; then
    echo "Rshare identified as remote."

    if rclone about rshare: 2>&1 | grep -q "Used:" ; then
        echo "Connected to remote rshare."
    else
        # check if password is obscured, obscure if not
        if rclone about rshare: 2>&1 | grep -q "couldn't decrypt password: base64 decode failed when revealing password - is it obscured?" ; then
            echo export RCLONE_CONFIG_RSHARE_PASS=$(rclone obscure $RCLONE_CONFIG_RSHARE_PASS) >> /root/.bashrc
            source /root/.bashrc
            echo "RCLONE: Password obscured to set up rshare"
        fi
        # check for error due to rclone version being higher than 1.62.2, amend endpoint if required
        if rclone about rshare: 2>&1 | grep -q "use the /dav/files/USER endpoint instead of /webdav" ; then
            echo "RCLONE running with a higher version than 1.62. Overwriting /webdav endpoint with /dav/files/USER to fix this..."
            echo export RCLONE_CONFIG_RSHARE_URL=${RCLONE_CONFIG_RSHARE_URL/webdav\/}dav/files/${RCLONE_CONFIG_RSHARE_USER} >> /root/.bashrc
            source /root/.bashrc
            echo "RCLONE: Overwrote /webdav endpoint with /dav/files/USER to fix RCLONE running with a higher version than 1.62."
        fi
        # finally, make sure we now have access to our storage!
        if ! rclone about rshare: 2>&1 | grep -q "Used:" ; then
            echo "Error in connecting to remote rshare."; sleep 3
            if [ "$0" != "$BASH_SOURCE" ]; then
                return 1
            else
                exit 1
            fi
        fi
        echo "Successfully connected to remote rshare."
    fi
else
    echo "Rshare not identified as (only) remote. Try to solve manually with AI4EOSC documentation."; sleep 5
    if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
    else
        exit 1
    fi
fi
echo "-----------------------"

# ########## Install prerequisites for deep-start
echo "Updating system..."
apt-get update
echo "Installing libgl1..."
apt-get install -y libgl1
echo "-----------------------"

# ########## Setup up API and submodule repositories
# upgrade pip
pip install --upgrade pip

# clone API repository
git clone --depth 1 https://github.com/emvollmer/tufsegm_api.git

# get submodule
cd tufsegm_api && git submodule update --init --recursive --remote

# install pre-requisites for repo installations
pip install packaging==22.0
# install repos
pip install -e ./TUFSeg/
pip install -e .

echo "================================================================"
echo "DEPLOYMENT SETUP COMPLETE"
echo "================================================================"

echo "Starting deepaas"
deep-start
