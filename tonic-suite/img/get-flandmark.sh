# library needed for (optional) facial alignment preprocessing step
# download Flandmark library and install

wget http://cmp.felk.cvut.cz/~uricamic/flandmark/ccount/click.php?id=7 -O flandmark.zip
unzip flandmark.zip
cd flandmark-master
cmake .
make
cp libflandmark/libflandmark_static.a ../
