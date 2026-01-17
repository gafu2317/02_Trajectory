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

*   [x] 模擬議論ログを用いた「脱線」の定義とアノテーション作成
*   [x] Embedding + LLMによる脱線判定精度の検証（F1スコア計測）
*   [x] 判定ロジックの固定（プロンプトエンジニアリング含む）

### Phase 2: プロトタイプ実装 (Streamlit)

*   [ ] Streamlitアプリケーションの基本構造をセットアップする。
*   [ ] チャットUI (`st.chat_message`) を実装する。
*   [ ] ユーザー入力 (議論のゴールと発言) を受け付ける機能を追加する。
*   [ ] リアルタイム埋め込み処理を`DriftDetector`クラスを通して統合する。
*   [ ] 類似度とLLMによる脱線判定結果をUIに表示する。
*   [ ] 簡易的な色分け表示（可視化Lv.1）を実装する。

