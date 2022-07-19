#!/bin/bash -f

x=`pwd`
echo "echo -e \"\033[0;31mRunning subshell...\" " > .subshell_initfile
cat .subshell_rc >> .subshell_initfile
#echo "export PS1=\"project_shell$ \" " >> .subshell_initfile
bash -c "export PYTHONPATH=$x; exec bash --init-file .subshell_initfile"
