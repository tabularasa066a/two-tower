# Two-Towerモデルを用いたレコメンデーションエンジンの構築
## Two-Towerモデルを使ったレコメンデーションエンジンの構築
### 実装参考
[参考](https://note.com/kokopelli_inc/n/nd89c1b89b741)

### その他参考
- ANN（近似最近傍探索）
- [Embedding（埋め込み）](https://zenn.dev/peishim/articles/c696ff85a539bd)
- [PyTorchにおける.train(),.eval()メソッドの使い方](https://nikkie-ftnext.hatenablog.com/entry/what-are-torch-nn-module-train-method-and-eval-method)

## 実行方法
```bash
# 以下３コマンドは初回のみ
pytohn3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 仮想環境有効化
source .venv/bin/activate
# 学習実行
python train.py
# 推論実行
python recommend.py
# 評価
python evaluate.py
```

## Python環境構築覚書
### 環境作成
```bash
# `.venv`という名称の仮想環境を構築
pytohn3 -m venv .venv
```

### 環境の有効化
```bash
# 有効化
source .venv/bin/activate

# 無効化
deactivate
```

## ライブラリ群のインストール
```bash
pip install -r requirements.txt
```

### pipインストールしたもの一覧をrequirements.txtに吐き出す
```bash
pip freeze > requirements.txt
```

### WSL上でグラフ表示できるようにする
```bash
sudo apt install imagemagick python3-tk -y
```
[参考](https://touch-sp.hatenablog.com/entry/2021/04/23/082702)

### CSVの行数をカウント
```bash
wc -l hoge.csv
```

### プロジェクトディレクトリ構成
```bash
.
│
├── train.py                    # 学習・モデル生成実行
├── recommend.py                # 推論実行
├── evaluate.py                 # 推論結果の評価
├── mlp_tower.py                # Two-Towerで用いるDNN部を定義
├── two_tower.py                # Two-Towerモデル本体
├── movie_lens_dataset.py       # MovieLensのデータセットをPyTorchで扱えるよう変換するクラス
├── test_pytorch.py             # PyTorchの動作確認用 
├── reset.sh                    # 学習済みデータを削除 
├── data/
│   ├── raw/
│   │    └── movielens-1m/      # 生データ（事前にダウンロードして配置）
│   │         ├── movies.dat
│   │         ├── ratings.dat
│   │         └── users.dat
│   └── processed/              
|        └── movielens-1m/      # 前処理済みデータの保存先
└── models/                     # モデルの保存先
    └── embeddings/             # 埋め込みベクトルの保存先
```

### データセットの構造
#### ユーザ一覧
```bash
# users.dat
UserID::Gender::Age::Occupation::Zip-code
```
↓（前処理で変換）↓
```bash
# users.csv
user_id,gender,age,occupation
```

#### 映画一覧
```bash
# movies.dat
MovieID::Title::Genres
```
↓（前処理で変換）↓
```bash
# items.csv
item_id,title,genre_Action,genre_Adventure,genre_Animation,genre_Children's,genre_Comedy,genre_Crime,genre_Documentary,genre_Drama,genre_Fantasy,genre_Film-Noir,genre_Horror,genre_Musical,genre_Mystery,genre_Romance,genre_Sci-Fi,genre_Thriller,genre_War,genre_Western
```

#### ユーザー映画評価一覧
```bash
# ratings.dat
UserID::MovieID::Rating::Timestamp
```
↓（前処理で変換）↓
```bash
# interactions.csv
user_id,item_id,rating,timestamp,gender,age,occupation,genre_Action,genre_Adventure,genre_Animation,genre_Children's,genre_Comedy,genre_Crime,genre_Documentary,genre_Drama,genre_Fantasy,genre_Film-Noir,genre_Horror,genre_Musical,genre_Mystery,genre_Romance,genre_Sci-Fi,genre_Thriller,genre_War,genre_Western
```