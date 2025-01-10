# Two-Towerモデルを用いたレコメンデーションエンジンの構築
## Two-Towerモデル
### 参考記事
[参考](https://note.com/kokopelli_inc/n/nd89c1b89b741)

## Python仮想環境の構築
### 環境作成
```bash
# .venvという名称の仮想環境を構築
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
pip install numpy scikit-learn jupyter
```

### pipインストールしたもの一覧をrequirements.txtに吐き出す
```bash
pip freeze > requirements.txt
```

### プロジェクトディレクトリ構成
```bash
.
├── train.py
├── recommend.py
├── evaluate.py
├── mlp_tower.py
├── two_tower.py
├── test_pytorch.py             # PyTorchの動作確認用 
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

### 生データセットの構造
#### users.dat
```bash
UserID::Gender::Age::Occupation::Zip-code
```

#### movies.dat
```bash
MovieID::Title::Genres
```

#### ratings.dat
```bash
UserID::MovieID::Rating::Timestamp
```
