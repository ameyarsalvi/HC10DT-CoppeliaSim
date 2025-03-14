#!/usr/bin/env bash

#cd /home/pkorrap/Downloads/CoppeliaSim/

cd $COPPELIASIM_ROOT_DIR

for i in {4..34..2}
#for i in 4
do
    konsole --noclose --new-tab -e ./coppeliaSim.sh -H -GzmqRemoteApi.rpcPort=$((23000+i)) -GwsRemoteApi.port=$((23050+1+i)) //home/pkorrap/Projects/HC10DT-CoppeliaSim/HC10DT.ttt && /bin/bash &
#    konsole --noclose --new-tab -e ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=$((23000+i)) -GwsRemoteApi.port=$((23050+1+i)) //home/pkorrap/Projects/HC10DT-CoppeliaSim/HC10DT.ttt && /bin/bash &
    sleep 2
done
