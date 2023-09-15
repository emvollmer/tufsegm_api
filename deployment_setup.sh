#!/bin/bash

# ########## Connect remotely to NextCloud (rshare)
if rclone listremotes | grep -q "rshare:" ; then
    echo "Rshare identified as remote."

    if rclone about rshare: 2>&1 | grep -q "Used:" ; then
        echo "Connected to remote rshare."
    else
        echo "Password needs to be obscured to set up rshare..."
        echo export RCLONE_CONFIG_RSHARE_PASS=$(rclone obscure $RCLONE_CONFIG_RSHARE_PASS) >> /root/.bashrc
        source /root/.bashrc
        if ! rclone about rshare: 2>&1 | grep -q "Used:" ; then
            echo "Error in connecting to remote rshare."
            if [ "$0" != "$BASH_SOURCE" ]; then
                return 1
            else
                exit 1
            fi
        fi
        echo "Connected to remote rshare."
    fi
else
    echo "Rshare not identified as (only) remote. Try to solve manually with AI4EOSC documentation."
    if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
    else
        exit 1
    fi
fi

api_name="tufsegm_api"
submod_name="ThermUrbanFeatSegm"

# ########## Clone API repository
if [[ $(pwd) != *$api_name && ! -d $api_name ]]; then
    echo "Cloning API repository."
    git clone --recurse-submodules https://github.com/emvollmer/$api_name.git
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
if ! grep -q "branch = master" ".gitmodules"; then
    echo -e "\tbranch = master" >> ".gitmodules"
    echo "Added 'branch = master' to .gitmodules."
fi
git submodule update --remote --recursive

# ########## Check current python version
required_ver="3.8"
current_ver=$(python --version 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
if [[ "$required_ver" == "$current_ver" ]]; then
    echo "$what version requirement $required_ver satisfied."
else
    echo "Provided $what version $current_ver is not compatible with required ${required_ver}).
    Please install $what $required_ver "; sleep 5
    if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
    else
        exit 1
    fi
fi

# ########## Install prerequisites for python3.8 venv creation
if ! dpkg -l | grep -q python3.8-venv; then
    echo "Installing necessary Python3.8-venv..."
    yes | add-apt-repository ppa:deadsnakes/ppa
    apt-get update
    yes | apt-get install python3.8-venv
fi

# ########## Check for activated venv, create one if there is none, otherwise activate it
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Currently in an activated virtual environment. Setup may proceed."
else
    echo "Currently not in an activated virtual environment. Must have one up to proceed with package installation..."
    venv_name="venv"
    venv_act="$venv_name"/bin/activate
    if test -f $venv_act; then
        echo "Virtual environment at '$venv_name' already exists. Activating..."
    else
        if python3.8 -m venv "$venv_name"; then
            echo "Virtual environment successfully created. Activating..."
        else
            echo "Python3.8 venv creation unsuccessful. Stopping...";
            if [ "$0" != "$BASH_SOURCE" ]; then
                return 1
            else
                exit 1
            fi
        fi
    fi
    source $venv_act
fi


# ########## Install submodule and api as editable
pip3 install -U pip

echo "Installing $submod_name repository as editable..."
pip3 install -e ./$submod_name

echo "Installing $api_name repository as editable..."
pip3 install -e .

deactivate

echo "================================================================"
echo "INITIAL DEPLOYMENT SETUP COMPLETE"
echo "================================================================"