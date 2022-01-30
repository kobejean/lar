# LocalizeAR

![LAR Test Status](https://github.com/kobejean/lar/actions/workflows/test.yml/badge.svg?branch=main)

LocalizeARのC＋＋ライブラリーです。

# 仕組み

**STEP　１**

様々な角度から写真を撮ってスキャンする。

![Scan Images](/docs/media/scan_images.jpeg)

**STEP　２**

特徴点抽出とマッチング

**STEP　３**

特徴点のマッチングを使ってグラフを作る。

![Graph Before Optimization](/docs/media/construct_graph.jpeg)

**STEP　４**

グラフの最適化

![Graph After Optimization](/docs/media/optimize_graph.jpeg)

