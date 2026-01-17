import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAIクライアントの初期化
# 環境変数 OPENAI_API_KEY が設定されている必要があります
client = OpenAI()

# --- 定数 ---
EMBEDDING_MODEL = "text-embedding-3-small"

# --- 関数 ---

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """指定されたテキストの埋め込みベクトルを取得する"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """2つのベクトル間のコサイン類似度を計算する"""
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def load_discussion(file_path: str) -> dict:
    """JSONファイルから議論データを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- メイン処理 ---

def main():
    """
    模擬議論データを読み込み、各発言とゴールとの類似度を計算して表示する。
    """
    print("--- 脱線検知ロジックの検証開始 ---")

    # 1. データの読み込み
    try:
        discussion_data = load_discussion("data/mock_discussion_01.json")
        goal = discussion_data["goal"]
        utterances = discussion_data["utterances"]
        print(f"議論のゴール: {goal}\n")
    except FileNotFoundError:
        print("エラー: `data/mock_discussion_01.json` が見つかりません。")
        return
    except json.JSONDecodeError:
        print("エラー: `data/mock_discussion_01.json` の形式が正しくありません。")
        return

    # 2. ゴールのベクトル化
    try:
        goal_embedding = get_embedding(goal)
        print("ゴールのベクトル化に成功しました。")
    except Exception as e:
        print(f"エラー: OpenAI APIへの接続に失敗しました。APIキーが正しく設定されているか確認してください。")
        print(f"詳細: {e}")
        return

    print("\n--- 各発言とゴールとのコサイン類似度 ---")
    # 3. 各発言をループし、類似度を計算
    for utterance in utterances:
        speaker = utterance["speaker"]
        text = utterance["text"]
        is_drift_label = utterance["is_drift"]

        # 発言のベクトル化
        utterance_embedding = get_embedding(text)

        # 類似度の計算
        similarity = cosine_similarity(goal_embedding, utterance_embedding)

        print(f"[{speaker}] {text}")
        print(f"  類似度: {similarity:.4f} (手動ラベル: {'脱線' if is_drift_label else '本筋'})")
        print("-" * 20)

if __name__ == "__main__":
    main()
