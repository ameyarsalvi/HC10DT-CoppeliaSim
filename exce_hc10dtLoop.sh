#!/usr/bin/env bash

cd /home/asalvi/CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu20_04/

for i in {4..6..2}
#for i in 4
do
    konsole --noclose --new-tab -e ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=$((23000+i)) -GwsRemoteApi.port=$((23050+1+i)) //home/asalvi/code_ws/hc10dt/HC10DT.ttt && /bin/bash &
    sleep 5
done