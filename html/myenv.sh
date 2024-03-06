#In bash, sh, or ksh, use these commands:
export NVHPC_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/24.1
export PATH=${NVHPC_DIR}/compilers/bin:$PATH
export MANPATH=${NVHPC_DIR}/compilers/man:$MANPATH

#To use MPI, also set:
export PATH=${NVHPC_DIR}/comm_libs/mpi/bin:$PATH

source /home/liboyang/CDIxx/env.sh
