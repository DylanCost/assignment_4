#!/bin/bash
mkdir -p data
cd data
#wget --no-check-certificate "https://www.crcv.ucf.edu/data/UCF50.rar"
#unar -f data/UCF50.rar -o data/
unrar x /content/data/UCF50.rar -d /content/data/
cd ..