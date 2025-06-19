#!/bin/bash
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
export CDI_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
if ! [[ $PATH =~ $CDI_DIR ]] ; then
  export PATH+=:$CDI_DIR/bin:$CDI_DIR/script:$CDI_DIR/python
fi
if ! [[ $PYTHONPATH =~ $CDI_DIR ]] ; then
  export PYTHONPATH+=:$CDI_DIR/lib:$CDI_DIR/python
fi
if [ $(uname) = "Darwin" ]; then
  if ! [[ $DYLD_LIBRARY_PATH =~ $CDI_DIR ]] ; then
    export DYLD_LIBRARY_PATH+=:$CDI_DIR/lib
  fi
else
  if ! [[ $LD_LIBRARY_PATH =~ $CDI_DIR ]] ; then
    export LD_LIBRARY_PATH+=:$CDI_DIR/lib
  fi
fi
alias cdicd='cd $CDI_DIR'
alias cdimake='ninja -C $CDI_DIR/build'
alias cdiremake='ninja -C $CDI_DIR/build clean; cmake -S $CDI_DIR -B $CDI_DIR/build --fresh; ninja -C $CDI_DIR/build'
