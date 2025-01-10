import torch
import torch.nn as nn

class MLPTower(nn.Module):
    """
    カテゴリカルな特徴量と数値特徴量を受け取り、
    固定次元の埋め込みベクトルを出力するMLP（多層パーセプトロン）モデル
    """

    def __init__(
        self,
        categorical_dims: list[int],
        n_numerical_features: int,
        embedding_dim: int = 8,
        hidden_dims: list[int] = [64, 32],
        output_dim: int = 32,
    ) -> None:
        super().__init__()

        # カテゴリカル変数の埋め込み層
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, embedding_dim) for dim in categorical_dims]
        )

        # 入力次元の計算
        total_embedding_dim = len(categorical_dims) * embedding_dim
        input_dim = total_embedding_dim + n_numerical_features

        # MLPレイヤーの構築
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        categorical_features: torch.Tensor | None,
        numerical_features: torch.Tensor,
    ) -> torch.Tensor:
        # カテゴリカル特徴量の埋め込み
        if categorical_features is not None and len(self.embeddings) > 0:
            embedded = [
                embedding(categorical_features[:, i])
                for i, embedding in enumerate(self.embeddings)
            ]
            embedded = torch.cat(embedded, dim=1)
            x = torch.cat([embedded, numerical_features], dim=1)
        else:
            x = numerical_features

        return self.mlp(x)