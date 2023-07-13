#!/bin/bash
source ../env.sh
cd $1
echo processimg_cu $2 $3 $4 $5 $6 $7 $8
processimg_cu $2 $3 $4 $5 $6 $7 $8 2>&1
