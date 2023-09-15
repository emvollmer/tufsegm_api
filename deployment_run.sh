#!/bin/bash

api_name="tufsegm_api"
submod_name="ThermUrbanFeatSegm"

# ########## Ensure API directory given (deployment_setup.sh was previously run)
if [[ $(pwd) != *$api_name && ! -d $api_name ]]; then
    echo "API repository not yet set up. Please check your directory or run the deployment_setup.sh script."
    if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
    else
        exit 1
    fi
fi
if [[ $(pwd) != *$api_name && -d $api_name ]]; then
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

# ########## Check for activated venv, create one if there is none, otherwise activate it
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Currently in an activated virtual environment. Setup may proceed."
else
    echo "Currently not in an activated virtual environment. Must have one to proceed..."
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

deepaas-run --listen-ip 0.0.0.0