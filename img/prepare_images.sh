ls input/$1/* > input/$1_list.txt
sed -i -e "s/$/ 0/" input/$1_list.txt
