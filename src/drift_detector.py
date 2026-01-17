import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

class DriftDetector:
    """
    議論の脱線を検知する機能を提供するクラス。
    """
    def __init__(self,
                 embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-4o-mini",
                 similarity_threshold: float = 0.25):
        """
        DriftDetectorを初期化する。

        Args:
            embedding_model (str): 埋め込みに使用するモデル名。
            llm_model (str): 判定に使用するLLMのモデル名。
            similarity_threshold (float): これを下回るとLLM判定をトリガーする類似度の閾値。
        """
        load_dotenv()
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.similarity_threshold = similarity_threshold

    def get_embedding(self, text: str) -> list[float]:
        """指定されたテキストの埋め込みベクトルを取得する"""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.embedding_model)
        return response.data[0].embedding

    def cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """2つのベクトル間のコサイン類似度を計算する"""
        vec1 = np.array(v1)
        vec2 = np.array(v2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def is_valid_diversion(self, goal: str, utterance: str) -> str:
        """発言がゴールに対して有効な多様化か、単なるノイズかをLLMに判定させる。"""
        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "あなたは議論を分析する優秀なアシスタントです。"
                            "議論のゴールと、それに対して関連性が低いと判断された発言が与えられます。"
                            "その発言が、議論の新たな側面を探る「意味のある逸脱」なのか、"
                            "あるいは単に関係のない「ノイズ」なのかを判定してください。"
                            "回答は「意味のある逸脱」または「ノイズ」のいずれかのみで返してください。"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"議論のゴール: 「{goal}」\n\n発言: 「{utterance}」"
                    }
                ],
                temperature=0,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"判定不能({e})"

    def analyze(self, discussion_data: dict) -> list[dict]:
        """
        議論データを分析し、各発言の類似度と脱線判定結果を返す。

        Args:
            discussion_data (dict): 'goal'と'utterances'のキーを持つ議論データ。

        Returns:
            list[dict]: 各発言の分析結果のリスト。
        """
        results = []
        goal = discussion_data["goal"]
        utterances = discussion_data["utterances"]

        try:
            goal_embedding = self.get_embedding(goal)
        except Exception as e:
            raise ConnectionError(f"OpenAI APIへの接続に失敗しました。APIキーを確認してください。詳細: {e}")

        for utterance in utterances:
            text = utterance["text"]
            utterance_embedding = self.get_embedding(text)
            similarity = self.cosine_similarity(goal_embedding, utterance_embedding)

            result = {
                "speaker": utterance["speaker"],
                "text": text,
                "is_drift_label": utterance.get("is_drift"),
                "similarity": similarity,
                "llm_judgment": None
            }

            if similarity < self.similarity_threshold:
                llm_judgment = self.is_valid_diversion(goal, text)
                result["llm_judgment"] = llm_judgment
            
            results.append(result)
            
        return results
