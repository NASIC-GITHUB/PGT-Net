#!/bin/bash
#### Author: Eric Shih
#### Email: eric12345566@gmail.com
#### Usage: 
####    $ ./create_folder.sh  <folder_name>


echo "[PGT-Net Utiles - Creating Training Folder Tool]"
echo ""
echo "Folder Name: $1"
echo ""

if [ -z "$1" ]
then
    echo "Parameter is empty, please use $0 <folder_name>"
    exit 1
fi

echo "[Basic Folder Check]"
if ! [ -d "./result" ]
then
    echo "> Result folder does not exist, creating..."
    mkdir ./result
else
    echo "> Result folder check ok! "
fi

if ! [ -d "./testing" ]
then
    echo "> Testing folder does not exist, creating..."
    mkdir ./testing 
else
    echo "> Testing folder check ok! "
fi

echo ''
echo '----------------------------'
echo ''
echo "[Create Training Folder]"

if [ -d "./result/$1" ] || [ -d "./testing/$1" ]
then
    echo "> $1 Exist in result or testing folder, exit..."
    exit 1
fi

# Creating Folder
mkdir ./result/$1
mkdir ./testing/$1
mkdir ./testing/$1/binary
mkdir ./testing/$1/non_binary

echo "> Done!"


