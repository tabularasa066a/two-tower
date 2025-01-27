import json
from datetime import datetime
from pathlib import Path

# NOTE: torch -> sklearnの順番でimportしないとエラーになる場合があります。
# ref: https://github.com/pytorch/pytorch/issues/31409
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import faiss
import numpy as np
import pandas as pd
## TQDM（プログレスバーの表示）はjupyter使う際は下のほうを代わりにコメントイン
from tqdm import tqdm
# from tqdm.notebook import tqdm

# 別ファイルに（two_tower.py）に定義したTwoTowerModelクラスをimport
from two_tower import TwoTowerModel
from mlp_tower import MLPTower
from movie_lens_dataset import MovieLensDataset

def preprocess_movielens_data(
        path_raw: Path, path_processed: Path
    ) -> None:
    """MovieLensデータセットを読み込み、前処理を行い、保存する"""

    print("Data preprocessing in progress...")
    # データ読み込み
    df_users = pd.read_csv(
        path_raw / "users.dat",
        sep="::",
        header=None,
        encoding="ISO-8859-1",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    ).drop(columns=["zip_code"])

    df_items = pd.read_csv(
        path_raw / "movies.dat",
        sep="::",
        header=None,
        encoding="ISO-8859-1",
        engine="python",
        names=["item_id", "title", "genres"],
    )

    df_ratings = pd.read_csv(
        path_raw / "ratings.dat",
        sep="::",
        header=None,
        encoding="ISO-8859-1",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    # ジャンルのone-hotエンコーディング
    genres_encoded = df_items["genres"].str.get_dummies(sep="|")
    genres_encoded.columns = [f"genre_{col}" for col in genres_encoded.columns]

    # 映画データの処理
    df_items = pd.concat(
        [df_items[["item_id", "title"]], genres_encoded],
        axis=1
    )

    # 性別を数値エンコーディング
    df_users["gender"] = df_users["gender"].map({"F": 1, "M": 2})

    # データの結合
    df_interactions = (
        df_ratings
        .merge(
            df_users[["user_id", "gender", "age", "occupation"]],
            on="user_id",
            how="left"
        )
        .merge(
            df_items,
            on="item_id",
            how="left"
        )
    ).drop(columns=["title"])

    # データの分割とシャッフル
    df_interactions_train, df_interactions_test = train_test_split(df_interactions, test_size=0.2, random_state=42)

    # 前処理済みデータの保存
    path_processed.mkdir(parents=True, exist_ok=True)
    df_users.to_csv(path_processed / "users.csv", index=False)
    df_items.to_csv(path_processed / "items.csv", index=False)
    df_interactions_train.to_csv(path_processed / "interactions_train.csv", index=False)
    df_interactions_test.to_csv(path_processed / "interactions_test.csv", index=False)

    print(f"\nData saved: {path_processed / 'users.csv'}")
    print(f"Data saved: {path_processed / 'items.csv'}")
    print(f"Data saved: {path_processed / 'interactions_train.csv'}")
    print(f"Data saved: {path_processed / 'interactions_test.csv'}")

def train_model(
    model: TwoTowerModel,
    dataset: Dataset,
    n_epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    save_path: str | Path = "./models",
) -> dict[str, list[float]]:
    """モデルの学習を実行"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    history = {"loss": []}

    best_loss = float("inf")

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="Training", position=1, leave=True):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(batch)
            loss = model.compute_loss(outputs["logits"], batch["rating"])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        history["loss"].append(avg_loss)
        print(f"\nEpoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }

        if avg_loss < best_loss:
            best_loss = avg_loss

            # 最良モデルの保存
            torch.save(checkpoint, save_path / "best_model.pth")

            # メタ情報の保存
            meta_info = {
                "best_epoch": epoch + 1,
                "best_loss": float(best_loss),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
            with open(save_path / "best_model_info.json", "w") as f:
                json.dump(meta_info, f, indent=2)

        # 最新のモデルの保存
        torch.save(checkpoint, save_path / f"model_epoch_{epoch+1:04d}.pth")

    return history


def load_checkpoint(
    model: TwoTowerModel,
    checkpoint_path: str | Path | None = None,
    model_path: str | Path = "./models",
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> tuple[TwoTowerModel, dict]:
    """保存されたモデルのチェックポイントを読み込む"""
    if checkpoint_path is None:
        checkpoint_path = Path(model_path) / "best_model.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, checkpoint

def calculate_embeddings(
    model: TwoTowerModel,
    df_users: pd.DataFrame,
    df_items: pd.DataFrame,
    save_path: Path,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """
    全ユーザーと全アイテムの埋め込みを計算する

    Args:
        model: 学習済みのTwoTowerModel
        df_users: ユーザー情報のDataFrame
        df_items: アイテム情報のDataFrame
        save_path: 埋め込みの保存先ディレクトリ
        device: 計算に使用するデバイス
        batch_size: バッチサイズ
    """
    print("埋め込みの計算を開始します...")
    model.eval()
    model.to(device)
    save_path.mkdir(parents=True, exist_ok=True)

    # ユーザー特徴量の準備
    user_categorical = torch.tensor(
        df_users[["gender", "occupation"]].values, dtype=torch.int32, device=device
    )
    user_numerical = torch.tensor(
        df_users[["age"]].values, dtype=torch.float32, device=device
    )

    # アイテム特徴量の準備
    genre_cols = [col for col in df_items.columns if col.startswith("genre_")]
    item_numerical = torch.tensor(
        df_items[genre_cols].values, dtype=torch.float32, device=device
    )

    with torch.no_grad():
        # ユーザー埋め込みの計算
        user_embeddings_list = []
        for i in tqdm(
            range(0, len(df_users), batch_size), desc="ユーザー埋め込みを計算中"
        ):
            batch_cat = user_categorical[i : i + batch_size]
            batch_num = user_numerical[i : i + batch_size]
            embeddings = model.user_tower(batch_cat, batch_num)
            embeddings = F.normalize(embeddings, p=2, dim=1)  # L2正規化
            user_embeddings_list.append(embeddings.cpu())

        user_embeddings = torch.cat(user_embeddings_list, dim=0)

        # アイテム埋め込みの計算
        item_embeddings_list = []
        for i in tqdm(
            range(0, len(df_items), batch_size), desc="アイテム埋め込みを計算中"
        ):
            batch_num = item_numerical[i : i + batch_size]
            embeddings = model.item_tower(None, batch_num)
            embeddings = F.normalize(embeddings, p=2, dim=1)  # L2正規化
            item_embeddings_list.append(embeddings.cpu())

        item_embeddings = torch.cat(item_embeddings_list, dim=0)

    return {"user_embeddings": user_embeddings, "item_embeddings": item_embeddings}


def load_embeddings(load_path: Path) -> dict[str, torch.Tensor]:
    user_embeddings = torch.load(load_path / "user_embeddings.pt", weights_only=True)
    item_embeddings = torch.load(load_path / "item_embeddings.pt", weights_only=True)
    return {"user_embeddings": user_embeddings, "item_embeddings": item_embeddings}


def build_faiss_index(embeddings: torch.Tensor) -> faiss.Index:
    """
    コサイン類似度を使用するFaissインデックスを構築する

    Args:
        embeddings: 埋め込みベクトルのリスト

    Returns:
        Faissのインデックス
    """
    embeddings_numpy = embeddings.numpy().astype(np.float32)
    dimension = embeddings_numpy.shape[1]

    # コサイン類似度用のインデックスを構築（内積用のインデックス）
    index = faiss.IndexFlatIP(dimension)

    # L2正規化を行う（コサイン類似度のため）
    faiss.normalize_L2(embeddings_numpy)
    index.add(embeddings_numpy)

    return index


def save_faiss_index(index: faiss.Index, save_path: Path) -> None:
    """
    Faissインデックスを保存する

    Args:
        index: 保存するFaissインデックス
        save_path: 保存先のパス
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(save_path))
    print(f"Faissインデックスを保存しました: {save_path}")


def load_faiss_index(load_path: Path) -> faiss.Index:
    """
    Faissインデックスを読み込む

    Args:
        load_path: 読み込むインデックスのパス

    Returns:
        読み込んだFaissインデックス
    """
    if not load_path.exists():
        raise FileNotFoundError(f"インデックスファイルが見つかりません: {load_path}")

    index = faiss.read_index(str(load_path))
    print(f"Faissインデックスを読み込みました: {load_path}")
    return index


if __name__ == "__main__":
    # データの読み込みと前処理
    path_raw = Path("./data/raw/movielens-1m")
    path_processed = Path("./data/processed/movielens-1m")
    preprocess_movielens_data(path_raw, path_processed)

    df = pd.read_csv(path_processed / "interactions_train.csv")

    dataset = MovieLensDataset(df)

    NUM_EPOCHS: int = 100
    BATCH_SIZE: int = 512

    # モデルの構築
    user_tower = MLPTower(
        categorical_dims=[3, 22],  # gender, occupation
        n_numerical_features=1,  # age
        embedding_dim=8,
        hidden_dims=[64, 32],
        output_dim=32,
    )
    item_tower = MLPTower(
        categorical_dims=[],  # アイテムにはカテゴリカル特徴量なし
        n_numerical_features=len(dataset.genre_columns),  # genres
        hidden_dims=[64, 32],
        output_dim=32,
    )
    model = TwoTowerModel(user_tower, item_tower)
    print(f"モデルパラメタ数：{len(list(model.parameters()))}")

    # チェックポイントの確認と読み込み
    checkpoint_path = Path("./models/best_model.pth")
    def get_device() -> str:
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    if checkpoint_path.exists():
        model, checkpoint = load_checkpoint(
            model=model, checkpoint_path=checkpoint_path, device=get_device()
        )
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")

    else:
        print("No checkpoint found. Starting from scratch.")


    # モデルの学習
    history = train_model(
        model=model,
        dataset=dataset,
        n_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        device=get_device(),
    )

    # 埋め込みの計算
    df_users = pd.read_csv(path_processed / "users.csv")
    df_items = pd.read_csv(path_processed / "items.csv")

    embeddings = calculate_embeddings(
        model=model,
        df_users=df_users,
        df_items=df_items,
        save_path=Path("./models/embeddings"),
        device=get_device(),
    )

    # Faissインデックスの構築
    item_index = build_faiss_index(embeddings["item_embeddings"])
    save_faiss_index(index=item_index, save_path=Path("./models/embeddings/item_embeddings.idx"))

    user_index = build_faiss_index(embeddings["user_embeddings"])
    save_faiss_index(index=user_index, save_path=Path("./models/embeddings/user_embeddings.idx"))
