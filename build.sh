cd third_party/g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
mkdir install
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make -j
make install
cd ../../../

mkdir build
cd build
cmake ..
make -j ba_demo
cd ../