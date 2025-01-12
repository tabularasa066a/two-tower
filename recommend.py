from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np

# 別ファイルに（two_tower.py）に定義したTwoTowerModelクラスをimport
from two_tower import TwoTowerModel
from mlp_tower import MLPTower
from train import load_checkpoint, load_faiss_index


def get_recommendations(
    user_data: dict,
    model: TwoTowerModel,
    item_index: faiss.Index,
    device: str = "cpu",
    n_recommendations: int = 10,
) -> tuple[list[int], list[float]]:
    """
    ユーザーのJSONデータからレコメンドを生成する

    Args:
        user_data: {
            "gender": "F" or "M",
            "age": int,
            "occupation": int (0-21)
        }
        model: 学習済みのTwoTowerModel
        item_index: アイテムのFaissインデックス
        device: 計算デバイス
        n_recommendations: 推薦アイテム数

    Returns:
        推薦アイテムのインデックスとスコアのタプル
    """
    # ユーザーデータの前処理
    gender_map = {"F": 1, "M": 2}
    categorical_features = torch.tensor(
        [[gender_map[user_data["gender"]], user_data["occupation"]]],
        dtype=torch.int32,
        device=device
    )
    numerical_features = torch.tensor(
        [[user_data["age"]]],
        dtype=torch.float32,
        device=device
    )

    # ユーザー埋め込みの計算
    model.eval()
    with torch.no_grad():
        user_embedding = model.user_tower(categorical_features, numerical_features)
        user_embedding = F.normalize(user_embedding, p=2, dim=1)

    # 最近傍検索
    user_vector = user_embedding.cpu().numpy().astype(np.float32)
    scores, neighbors = item_index.search(user_vector, n_recommendations)

    return neighbors[0].tolist(), scores[0].tolist()


if __name__ == "__main__":
    # モデルとインデックスの読み込み
    model_path = Path("./models/best_model.pth")
    index_path = Path("./models/embeddings/item_embeddings.idx")
    device = "cpu"

    # モデルの構築と読み込み
    user_tower = MLPTower(
        categorical_dims=[3, 22],  # gender, occupation
        n_numerical_features=1,    # age
        embedding_dim=8,
        hidden_dims=[64, 32],
        output_dim=32,
    )
    item_tower = MLPTower(
        categorical_dims=[],
        n_numerical_features=18,   # genres
        hidden_dims=[64, 32],
        output_dim=32,
    )
    model = TwoTowerModel(user_tower, item_tower)
    model, _ = load_checkpoint(model, model_path, device=device)

    # インデックスの読み込み
    item_index = load_faiss_index(index_path)

    # テスト用のユーザーデータ
    test_user = {
        "gender": "M",
        "age": 33,
        "occupation": 12  # 例：プログラマ
    }

    # レコメンド生成
    items, scores = get_recommendations(
        user_data=test_user,
        model=model,
        item_index=item_index,
        device=device,
        n_recommendations=5
    )

    # 結果の表示
    print("\nレコメンド結果:")
    print(f"ユーザー: {test_user}")
    print(f"推薦アイテムID: {items}")
    scores_rounded = [round(score, 4) for score in scores]
    print(f"類似度スコア: {scores_rounded}")