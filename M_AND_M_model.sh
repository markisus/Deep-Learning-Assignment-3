#!/bin/bash

read N

for((i=0;i<$N;i++))
    do
        read line
        th classify.lua -input "$line"
    done