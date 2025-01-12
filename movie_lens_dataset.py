import torch
from torch.utils.data import Dataset
import pandas as pd

class MovieLensDataset(Dataset):
    """MovieLensデータセット用のカスタムデータセット

    interactions.csvからユーザーとアイテムの特徴量、および評価値を保持し、
    バッチ学習用のテンソルを提供します。
    """
    def __init__(self, df: pd.DataFrame) -> None:
        self.length = len(df)
        self.genre_columns = [col for col in df.columns if col.startswith("genre_")]

        # 特徴量の定義
        self.user_categorical_columns = ["gender", "occupation"]
        self.user_numerical_columns = ["age"]
        self.item_numerical_columns = self.genre_columns

        # テンソルへの変換
        self.user_categorical_features = torch.tensor(
            df[self.user_categorical_columns].values, dtype=torch.int32
        )
        self.user_numerical_features = torch.tensor(
            df[self.user_numerical_columns].values, dtype=torch.float32
        )
        self.item_numerical_features = torch.tensor(
            df[self.item_numerical_columns].values, dtype=torch.float32
        )
        self.rating = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "user_categorical_features": self.user_categorical_features[idx],
            "user_numerical_features": self.user_numerical_features[idx],
            "item_numerical_features": self.item_numerical_features[idx],
            "rating": self.rating[idx],
        }