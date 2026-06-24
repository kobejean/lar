# LocalizeAR

![LAR Test Status](https://github.com/kobejean/lar/actions/workflows/test.yml/badge.svg?branch=main)

![LARExplore](/docs/media/lar_explore2.png)

## Demo

Watch the demo video: [LocalizeAR Demo on YouTube](https://www.youtube.com/watch?v=QvLbRKkttDE)

LocalizeARのC＋＋ライブラリーです。

Swift版はこちら　→　[lar-swift](https://github.com/kobejean/lar-swift)

## Map Reconstruction Pipeline

Build maps from LARScan captures using COLMAP/GLOMAP. The Python environment is
managed with [uv](https://docs.astral.sh/uv/) (the project standard).

- **[docs/INSTALLATION.md](docs/INSTALLATION.md)** — toolchain setup (uv, COLMAP, GLOMAP)
- **[docs/RECONSTRUCTION.md](docs/RECONSTRUCTION.md)** — capture with LARScan → transfer to Mac → reconstruct

Quick run (after setup):
```sh
uv run python script/colmap/colmap.py input/<session> --use_glomap
```

# コンパイルする方法

普通
```sh
make all
```

並列コンピューティングを活用する
```sh
make fast
```

Testを含める
```sh
make tests
```

XCFrameworkをコンパイル
```sh
make frameworks
```

