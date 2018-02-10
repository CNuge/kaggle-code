#!/bin/bash

files=$(ls *.csv)
for file in $files
do
	tail -n +1 $file >> ../alldat5yr.csv
done