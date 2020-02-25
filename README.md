**English version is available [here](https://jp.mathworks.com/matlabcentral/fileexchange/65286-heart-sound-classifier)**

[![View  Heart Sound Classifier: MATLAB による心音分類アプリケーション開発 on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://jp.mathworks.com/matlabcentral/fileexchange/70555-heart-sound-classifier-matlab)
# Introduction

eBook:「機械学習をマスターする: MATLAB ステップ・バイ・ステップ ガイド」で紹介するデモを再現するためのスクリプトです。

詳細解析は[こちら](https://github.com/mathworks/Heart-Sound-Classifier-Japanese-ver/blob/master/HeartSoundClassificationR2018b_JP/HeartSoundClassification_LiveScript_JP.md)


# What does this example provide

このデモでは、組み込み機械学習アプリ開発のための一連のワークフロー、具体的にはデータの読み込み、
特徴抽出、各種アルゴリズムの検討、モデルのチューニング、そしてプロトタイプ配布までを紹介します。

特にここでは、危険な心臓病のリスクがある患者を診察する医療業務で応用可能で、
臨床医への依存軽減につながる、心音の"正常"と"異常"を分類するアルゴリズムを開発します。

eBook は[こちら](https://jp.mathworks.com/campaigns/offers/mastering-machine-learning-with-matlab.html) から DL できます。

このアプリケーションの開発では、以下の手順に従います。
1. データの読み込みと探索
2. データを前処理して特徴抽出
3. 予測モデルを開発
4. モデルの最適化
5. 学習済みモデルのCコード生成


## Environment

MATLAB R2018b以降

使用する Toolbox
- Statistics and Machine Learning
- Signal Processing
- MATLAB Coder (C code generation part only)

## Note

すべての解析フローは HeartSoundClassification_LiveScript_JP.mlx で完結していますが、
HelperFunctions フォルダ内の関数も適宜使用しています。

また使用するデータセット（音声ファイル）は別途ダウンロードしていただく必要がありますが、
上記スクリプトの冒頭でダウンロードするようなプログラムとしています。

データ元： [PhysioNet/CinC challenge 2016](http://www.physionet.org/physiobank/database/challenge/2016)

音声ファイルから抽出した特徴量、アンサンブル決定木で学習した分類器などは MAT ファイルとして保存しています。

Copyright 2019-2020 The MathWorks, Inc.