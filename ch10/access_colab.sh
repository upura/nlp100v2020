#!/bin/bash

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open https://colab.research.google.com/drive/1PbT0co10CHFw12PiQTlC_91f0VggDCeX?authuser=1
  sleep 3600
done
