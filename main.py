import argparse
import json
from src.drift_detector import DriftDetector

def load_discussion(file_path: str) -> dict:
    """JSONファイルから議論データを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {file_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"エラー: ファイルのJSON形式が正しくありません: {file_path}")
        exit(1)

def print_evaluation_metrics(results: list[dict], threshold: float):
    """
    分析結果から評価指標（Precision, Recall, F1-score）を計算して表示する。
    """
    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative

    print("\n--- 評価 ---")
    print(f"閾値={threshold:.2f} での性能評価")

    for r in results:
        is_positive_label = r['is_drift_label'] is True
        # LLMの判定も加味した予測
        is_predicted_positive = (r['similarity'] < threshold and r['llm_judgment'] == 'ノイズ')

        if is_predicted_positive and is_positive_label:
            tp += 1
        elif is_predicted_positive and not is_positive_label:
            fp += 1
        elif not is_predicted_positive and is_positive_label:
            fn += 1

    # Precision（適合率）: 陽性と予測したうち、実際に陽性だった割合
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # Recall（再現率）: 実際の陽性のうち、陽性と予測できた割合
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # F1-score: PrecisionとRecallの調和平均
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("-" * 20)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1_score:.4f}")
    print("-" * 20)


def main():
    """
    コマンドラインから受け取ったファイルパスを元に、議論の脱線検知を実行する。
    """
    parser = argparse.ArgumentParser(description="議論の脱線を検知するツール")
    parser.add_argument("file_path", help="分析対象の議論データ(JSONファイル)のパス")
    args = parser.parse_args()

    # 1. データの読み込み
    print(f"'{args.file_path}' を読み込んでいます...")
    discussion_data = load_discussion(args.file_path)
    print(f"議論のゴール: {discussion_data.get('goal', 'N/A')}\n")

    # 2. 脱線検知器の初期化と実行
    try:
        detector = DriftDetector()
        print(f"--- 脱線検知を開始します (閾値: {detector.similarity_threshold}) ---")
        results = detector.analyze(discussion_data)
    except ConnectionError as e:
        print(f"エラー: {e}")
        exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        exit(1)

    # 3. 結果の表示
    for result in results:
        label = result['is_drift_label']
        label_text = 'N/A'
        if label is not None:
            label_text = '脱線' if label else '本筋'

        print(f"[{result['speaker']}] {result['text']}")
        print(f"  類似度: {result['similarity']:.4f} (手動ラベル: {label_text})")

        if result['llm_judgment']:
            print(f"  [類似度が低いため、LLMで判定...]")
            print(f"  LLMの判定: {result['llm_judgment']}")
        
        print("-" * 20)
    
    # 4. 評価指標の表示
    print_evaluation_metrics(results, detector.similarity_threshold)

if __name__ == "__main__":
    main()
