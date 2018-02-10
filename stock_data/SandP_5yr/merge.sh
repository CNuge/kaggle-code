#!/bin/bash

echo "date,open,high,low,close,volume,Name" > ../alldat5yr.csv

files=$(ls *.csv)
for file in $files
do
	tail -n +1 $file >> ../alldat5yr.csv
done