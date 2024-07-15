#!/bin/bash

# ########## Connect remotely to NextCloud (rshare)
if rclone listremotes | grep -q "rshare:" ; then
    echo "Rshare identified as remote."

    if rclone about rshare: 2>&1 | grep -q "Used:" ; then
        echo "Connected to remote rshare."
    else
        # check if password is obscured, obscure if not
        if rclone about rshare: 2>&1 | grep -q "couldn't decrypt password: base64 decode failed when revealing password - is it obscured?" ; then
            echo "Password needs to be obscured to set up rshare..."
            echo export RCLONE_CONFIG_RSHARE_PASS=$(rclone obscure $RCLONE_CONFIG_RSHARE_PASS) >> /root/.bashrc
            source /root/.bashrc
        fi
        # check for error due to rclone version being higher than 1.62.2, amend endpoint if required
        if rclone about rshare: 2>&1 | grep -q "use the /dav/files/USER endpoint instead of /webdav" ; then
            echo "RCLONE running with a higher version than 1.62. Overwriting /webdav endpoint with /dav/files/USER to fix this..."
            echo export RCLONE_CONFIG_RSHARE_URL=${RCLONE_CONFIG_RSHARE_URL/webdav\/}dav/files/${RCLONE_CONFIG_RSHARE_USER} >> /root/.bashrc
            source /root/.bashrc
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
        echo "Connected to remote rshare."
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

# ########## Check if Python 3.8, install if not
echo "Checking Python 3.8 installation"
if ! command -v python3.8 &>/dev/null; then
    echo "Python 3.8 is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install python3.8
    if ! command -v python3.8 &>/dev/null; then
        echo "Installation of Python3.8 failed. Please try manually!"; sleep 5
        if [ "$0" != "$BASH_SOURCE" ]; then
            return 1
        else
            exit 1
        fi
    fi
fi
# ########## Activate Python 3.8
if command -v python3.8 &>/dev/null; then
    alias python=python3.8
    echo "Python 3.8 is activated"
fi
echo "-----------------------"

# ########## Installing python-related prerequisites (necessary for deep-start to work)
echo "Installing required ppa:deadsnakes/ppa"
yes | add-apt-repository ppa:deadsnakes/ppa
echo "Finished installing ppa:deadsnakes/ppa."

echo "Installing libgl1..."
if ! apt-get install -y libgl1; then
    echo "Failed to install libgl1"
    sleep 5
fi
echo "Finished installing libgl1."

echo "Installing libgl1-mesa-glx..."
if ! apt-get install -y libgl1-mesa-glx; then
    echo "Failed to install libgl1-mesa-glx"
    sleep 5
fi
echo "Finished installing libgl1-mesa-glx."

# ########## Clone API repository
api_name="tufsegm_api"
submod_name="TUFSeg"

if [[ $(pwd) != *$api_name && ! -d $api_name ]]; then
    echo "-----------------------"
    echo "Cloning API repository."
    git clone --recurse-submodules https://github.com/emvollmer/$api_name.git   # with ssh key: git@github.com:emvollmer/tufsegm_api.git
    echo "Change into API directory"
    cd $api_name
elif [[ $(pwd) != *$api_name && -d $api_name ]]; then
    echo "Change into API directory."
    cd $api_name
fi

# ########## Update submodule
echo "Update submodule."
git pull --recurse-submodules
# add branch name of submodule to .gitmodules
if ! grep -q "branch = main" ".gitmodules"; then
    echo -e "\tbranch = main" >> ".gitmodules"
    echo "Added 'branch = main' to .gitmodules."
fi
git submodule update --remote --recursive

# ########## Install submodule and api as editable
pip3 install -U pip

echo "Installing $submod_name repository as editable..."
pip3 install -e ./$submod_name

echo "Installing $api_name repository as editable..."
pip3 install -e .

pip3 install -r requirements-mlflow.txt
#pip3 install -r requirements-test.txt

echo "================================================================"
echo "DEPLOYMENT SETUP COMPLETE"
echo "================================================================"

# ########## Make sure CUDA exists and add as PATH variable
local_cuda_path="/usr/local/cuda/"
if ls $local_cuda_path &> /dev/null ; then
    if [ -z "$CUDA_HOME" ] ; then
        export CUDA_HOME=$local_cuda_path
        echo "CUDA_HOME defined as $local_cuda_path"
    else
        echo "CUDA_HOME already defined as $CUDA_HOME"
    fi
else
    echo "CUDA does not exist at $local_cuda_path! Please check and define path manually..."
    if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
    else
        exit 1
    fi
fi

echo "Starting deepaas"
deepaas-start