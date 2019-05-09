#!/bin/bash

for i in $(seq 2001 2012);do
	ssconvert $i.xls $i.csv
done

for i in $(seq 2013 2018);do
	ssconvert $i.xlsx $i.csv
done