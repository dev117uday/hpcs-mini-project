#!/bin/bash

directory="/home/ubuntu/1800"


for file in "$directory"/*; do
  if [ -f "$file" ]; then
    ./opencvTest $file >> result2.txt
  fi
done