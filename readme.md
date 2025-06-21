## 環境
macbook air (m3)
Python 3.11.11

## 内容
研究で静的なハンドジェスチャ(指差しの方向)を推論するためのモデルを作成するコード

## 環境構築(pipenvを使用した場合)
このディレクトリで
```
pipenv install
```
data以下にtrain、valというディレクトリを作成。
trainには訓練用データ
valには検証用データを入れる。
データはフォルダ名をラベル名にする

- data
    - train
        - ラベル1
            - data1.png
            - data2.png
        - ラベル2
        - ラベル3
    - test
        - ラベル1
        - ラベル2
        - ラベル3

## 参考
背景の影響を減らすために以下を参考にしました
https://arxiv.org/abs/2009.10762