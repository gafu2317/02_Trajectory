# Gemini 開発メモ

このファイルは、AIアシスタントが "Trajectory" 開発プロジェクトを遂行する上での思考プロセス、計画、および進捗を記録するためのものです。

## 1. プロジェクト概要の理解

- **目的:** LLMとベクトル埋め込みを用いて議論の「軌跡」を可視化し、目的からの逸脱（脱線）をリアルタイムで検出・警告する。
- **対象:** 研究用途のlocalhost環境で動作するStreamlitアプリケーション。

## 2. 技術スタック

- **Language:** Python 3.10+
- **Frontend/App:** Streamlit
- **LLM:** OpenAI API (gpt-4o-mini)
- **Vector Store:** ChromaDB or FAISS (インメモリ)
- **Embedding:** OpenAI text-embedding-3-small
- **Visualization:** Plotly

## 3. 開発計画

`preStudy002.md`のフェーズに基づき、まずはCLIベースでのコアロジック検証から着手します。

### Phase 1: コアロジック検証 (CLIベース)

1.  **[ ] 環境構築:**
    -   Python仮想環境のセットアップ
    -   必要なライブラリ (`openai`, `numpy`, `scikit-learn`, `chromadb` or `faiss-cpu`) のインストール (`requirements.txt` の作成)
2.  **[ ] データ準備:**
    -   模擬的な議論ログ（テキストファイル or JSON）を作成する。議論のゴール、各発言、および「脱線」ラベルを含む。
3.  **[ ] 脱線判定ロジックの実装:**
    -   発言をEmbedding APIでベクトル化する関数を実装。
    -   ゴールと各発言のコサイン類似度を計算する関数を実装。
    -   類似度が閾値以下の場合に、gpt-4o-miniに「有効な転換点か、単なるノイズか」を判定させるプロンプトを作成・実装。
4.  **[ ] 評価:**
    -   準備したデータを用いて脱線判定の精度（Precision, Recall, F1スコア）を計測し、ロジック（閾値やプロンプト）を調整する。

---

まずは、**Phase 1-1: 環境構築** から開始します。

