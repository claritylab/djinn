#!/bin/bash

ls input/imc/* > input/imc_list.txt
sed -i -e "s/$/ 0/" input/imc_list.txt

ls input/face/* > input/face_list.txt
sed -i -e "s/$/ 0/" input/face_list.txt

ls input/dig/* > input/dig_list.txt
sed -i -e "s/$/ 0/" input/dig_list.txt
