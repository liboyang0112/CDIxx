#!/bin/bash
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
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

# Print all exported vars so fish can read them
echo "CDI_DIR=$CDI_DIR"
echo "PATH=$PATH"
if [[ -n "$PYTHONPATH" ]]; then
  echo "PYTHONPATH=$PYTHONPATH"
fi
if [[ -n "$DYLD_LIBRARY_PATH" ]]; then
  echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"
fi
if [[ -n "$LD_LIBRARY_PATH" ]]; then
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
fi
alias cdicd='cd $CDI_DIR'
alias cdimake='ninja -C $CDI_DIR/build'
alias cdiremake='ninja -C $CDI_DIR/build clean; cmake -S $CDI_DIR -B $CDI_DIR/build --fresh; ninja -C $CDI_DIR/build'
