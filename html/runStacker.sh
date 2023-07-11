#!/bin/bash
source ../env.sh
cd $1
echo processimg_cu $2 $3 $4 $5 $6
processimg_cu $2 $3 $4 $5 $6 2>&1
