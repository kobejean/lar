# `g2o_viewer`のインストール

`g2o_viewer`をインストールするには`Qt5`、`libQGLViewer`、`Eigen`が必用です。

## Install `Eigen`

```
brew install eigen
```

## Install `Qt5`

```
brew install qt@5
```

## Install `libQGLViewer`

Download and decompress: http://www.libqglviewer.com/src/libQGLViewer-2.7.2.tar.gz
```
cd libQGLViewer-2.7.2/QGLViewer
$(brew --prefix qt@5)/bin/qmake
make
sudo make install
```

## Install `g2o_viewer`

```
cd path/to/lar/thirdparty/g2o
mkdir build
cd build
cmake ..
make
make install
```