import torch
import torch.nn as nn
import torch.nn.functional as F

# 別ファイルに（mlp_tower.py）に定義したMLPTowerクラスをimport
from mlp_tower import MLPTower

class TwoTowerModel(nn.Module):
    """
    ユーザーとアイテムの特徴量を受け取り、
    ユーザーの埋め込みベクトルとアイテムの埋め込みベクトルを出力するTwo-Towerモデル
    """

    def __init__(
        self, user_tower: MLPTower, item_tower: MLPTower, temperature: float = 0.1
    ) -> None:
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # ユーザーとアイテムの埋め込みを計算
        user_embedding = self.user_tower(
            batch["user_categorical_features"], batch["user_numerical_features"]
        )
        item_embedding = self.item_tower(
            None,  # アイテムにはカテゴリカル特徴量なし
            batch["item_numerical_features"],
        )

        # L2正規化
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        item_embedding = F.normalize(item_embedding, p=2, dim=1)

        # コサイン類似度の計算
        logits = torch.matmul(user_embedding, item_embedding.t()) / self.temperature

        return {
            "logits": logits,
            "user_embedding": user_embedding,
            "item_embedding": item_embedding,
        }

    def compute_loss(
        self, logits: torch.Tensor, ratings: torch.Tensor, neutral_weight: float = 0.3
    ) -> torch.Tensor:
        """評価値に基づく損失を計算"""
        # 評価値を類似度スコアに変換
        mapping = {1: -1.0, 2: -0.5, 3: neutral_weight, 4: 0.7, 5: 1.0}
        similarity_targets = torch.tensor(
            [mapping[rate.item()] for rate in ratings], device=ratings.device
        )

        # 類似度損失の計算
        targets = (similarity_targets + 1) / 2
        return F.binary_cross_entropy_with_logits(
            logits, targets.unsqueeze(0).expand_as(logits)
        )