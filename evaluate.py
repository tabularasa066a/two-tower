from pathlib import Path


import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import faiss
from tqdm.notebook import tqdm

# 別ファイルに（two_tower.py）に定義したTwoTowerModelクラスをimport
from two_tower import TwoTowerModel
from mlp_tower import MLPTower
from recommend import get_recommendations
from train import load_checkpoint, load_faiss_index

def calculate_metrics_at_k(
    recommended_items: list[int],
    user_ratings: pd.DataFrame,
    k: int
) -> dict[str, float]:
    """
    推薦アイテムに対する各種評価指標を計算する

    Args:
        recommended_items: 推薦アイテムのリスト
        user_ratings: ユーザーの評価データ（item_id, ratingを含むDataFrame）
        k: 評価する推薦アイテム数

    Returns:
        各評価指標の値を含む辞書
    """
    if len(recommended_items) == 0:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0
        }

    # 実際に高評価したアイテム（rating >= 4）
    positive_items = user_ratings[user_ratings["rating"] >= 4]["item_id"].tolist()

    # 評価値の辞書を作成
    item_ratings = dict(zip(user_ratings["item_id"], user_ratings["rating"]))

    # 各指標の計算
    recommended_k = recommended_items[:k]
    n_relevant = sum(1 for item in recommended_k if item in positive_items)

    # Precision@K
    precision = n_relevant / k if k > 0 else 0.0

    # Recall@K
    recall = n_relevant / len(positive_items) if positive_items else 0.0

    # NDCG@K
    relevance_scores = [item_ratings.get(item, 0.0) for item in recommended_k]
    ideal_scores = sorted(item_ratings.values(), reverse=True)[:k]

    # スコアリストの長さを揃える
    relevance_scores.extend([0] * (k - len(relevance_scores)))
    ideal_scores.extend([0] * (k - len(ideal_scores)))

    ndcg = ndcg_score([ideal_scores], [relevance_scores])

    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"ndcg@{k}": ndcg
    }


def evaluate_model(
    model: TwoTowerModel,
    df_test: pd.DataFrame,
    item_index: faiss.Index,
    k: int = 10,
    device: str = "cpu"
) -> dict[str, float]:
    """
    テストデータセットに対してモデルの評価を行う

    Args:
        model: 評価するモデル
        df_test: テストデータ
        item_index: アイテムのFaissインデックス
        k: 評価する推薦アイテム数
        device: 計算デバイス

    Returns:
        評価指標の平均値を含む辞書
    """
    metrics_list = []
    model.eval()

    for user_id in tqdm(df_test["user_id"].unique(), desc="モデル評価中"):
        # ユーザーの評価データを取得
        user_ratings = df_test[df_test["user_id"] == user_id]

        # ユーザーデータの準備
        user_data = {
            "gender": "F" if user_ratings["gender"].iloc[0] == 1 else "M",
            "age": user_ratings["age"].iloc[0],
            "occupation": user_ratings["occupation"].iloc[0]
        }

        # 推薦アイテムを取得
        recommended_items, _ = get_recommendations(
            user_data=user_data,
            model=model,
            item_index=item_index,
            device=device,
            n_recommendations=k
        )

        # 評価指標の計算
        metrics = calculate_metrics_at_k(
            recommended_items=recommended_items,
            user_ratings=user_ratings,
            k=k
        )
        metrics_list.append(metrics)

    # 平均値の計算
    avg_metrics = {}
    for metric in metrics_list[0].keys():
        avg_metrics[metric] = np.mean([m[metric] for m in metrics_list])

    return avg_metrics


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

    # テストデータの読み込み
    path_processed = Path("./data/processed/movielens-1m")
    df_test = pd.read_csv(path_processed / "interactions_test.csv")

    # モデルの評価
    metrics = evaluate_model(
        model=model,
        df_test=df_test,
        item_index=item_index,
        k=10,
        device=device
    )

    # 結果の表示
    print("\n評価結果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")