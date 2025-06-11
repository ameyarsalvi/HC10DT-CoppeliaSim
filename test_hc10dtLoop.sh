#!/usr/bin/env bash

#cd /home/pkorrap/Downloads/CoppeliaSim/

cd $COPPELIASIM_ROOT_DIR

konsole --noclose --new-tab -e ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=$((23004)) -GwsRemoteApi.port=$((23055)) //home/pkorrap/Projects/HC10DT-CoppeliaSim/HC10DT.ttt && /bin/bash

#done

cd ~/Projects/HC10DT-CoppeliaSim
