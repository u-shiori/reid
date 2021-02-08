# 深層距離学習
## 種類
- Triplet, Improved Triplet, Quadruplet, ClusterTriplet
- Arcface, UIR



## 使用方法

### 0. セットアップ
```sh
git clone https://github.com/u-shiori/reid
cd reid
pip install -r requirements.txt
```

### 1. データセットの作成
```sh
sh generate_traindata.sh
```
### 入力

```sh
python src/train.py --model Triplet
                            Arcface
                            ImprovedTriplet
                            Quadruplet
                            ClusterTriplet
                            UIR
```



`train.py`内にあるパラメータを下記で説明。

##### model parameter

* margin_a: Triplet Loss 使用時のマージンα
* margin_b: ImprovedTriplet, Quadruplet 使用時のマージンβ

* w: UIR 使用時における半教師あり損失関数の重み




##### optuna parameter

* n_trials: optunaのトライアル数

* timeout: optunaのタイムアウト時間



##### train parameter

* n_epochs: エポック数

* log_interval: 学習時のログ間隔

* save_epoch_interval: 学習モデル保存の間隔

* in_channels: 学習時のチャネル数(RGBならば基本的に3で固定)

* lr: 学習率



##### dataset parameter

* cfg_train_path: 学習データのconfigファイルパス

* cfg_valid_path: 検証データのconfigファイルパス

* cfg_test_path: テストデータのconfigファイルパス



### 出力

```result/checkpoints/```に学習モデルが保存される。



## 注意点

* 学習に使うデータセットの形は`data/MOT20-03`の様な形で作成・配置していただけると学習ができる様になっています。
  この形のまま(プログラムの内容を変えずに)使用いただく場合、

  * `images/`の中に画像データ
  * `train.txt`, `test.txt`内の各行に`画像パス,ラベル`を記述したもの

  この２点を作成しますと、上記の`cfg_train_path`などでこのファイルを指定することで本学習のデータセットとして使用いただけます。


* `src/train.py`以外のスクリプトはそれぞれ機能別に分類しているだけなので、必要であれば適宜ご覧になってください。



## 参考サイト・文献

[1] [Github: Triplet Loss](https://github.com/adambielski/siamese-triplet)

[2] [Github: Arcface](https://github.com/ronghuaiyang/arcface-pytorch)

[3] J. Wang, Y. Song, T. Leung, C. Rosenberg, J. Wang, J. Philbin, B. Chen, and Y. Wu. "Learning Fine-grained Image Similarity with Deep Ranking." In CVPR, 2014.

[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," in rXiv:1801.07698, 2018.

[5] D. Cheng, Y. Gong, S. Zhou, J. Wang, and N. Zheng. "Person Re-Identification by Multi-Channel Parts-Based CNN with Improved Triplet Loss Function." In CVPR, 2016.

[6] W. Chen, X. Chen, J. Zhang, and K. Huang. "Beyond triplet loss: a deep quadruplet network for person re-identification." In CVPR, 2017.

[7] H. Yu, Y. Fan, K. Chen, H. Yan, X. Lu, J. Liu, and D. Xie. "Unknown Identity Rejection Loss: Utilizing Unlabeled Data for Face Recognition," 2019

