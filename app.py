import streamlit as st
import json
import glob
import textwrap
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from src.drift_detector import DriftDetector

# Streamlitページの基本設定
st.set_page_config(page_title="Trajectory: 議論脱線検知システム", layout="wide")
st.title("Trajectory: 議論脱線検知システム")

# --- 関数群 ---
@st.cache_data
def load_json_data(file_path: str) -> dict:
    """JSONファイルを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
        return None

def create_2d_plot(results: list[dict], goal_embedding: list[float], threshold: float):
    """PCAによる次元削減とPlotlyによる2Dマップ描画"""
    # ゴールと発言のベクトルをすべて集める
    embeddings = [goal_embedding] + [r['embedding'] for r in results]
    
    # PCAで2次元に削減
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    # データをPandas DataFrameに変換
    df_data = []
    # 0番目はゴール
    df_data.append({
        "speaker": "GOAL", 
        "text": "議論のゴール", 
        "similarity": 1.0, 
        "is_drift": False,
        "x": coords[0, 0],
        "y": coords[0, 1]
    })
    # 1番目以降は発言
    for i, r in enumerate(results):
        df_data.append({
            "speaker": r['speaker'],
            "text": r['text'],
            "similarity": r['similarity'],
            "is_drift": r['similarity'] < threshold,
            "x": coords[i+1, 0],
            "y": coords[i+1, 1]
        })
    df = pd.DataFrame(df_data)

    # ゴールを原点(0,0)に移動させる
    goal_x, goal_y = df.loc[df['speaker'] == 'GOAL', ['x', 'y']].iloc[0]
    df['x'] -= goal_x
    df['y'] -= goal_y
    
    # Plotlyで散布図を作成
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='is_drift',
        color_discrete_map={True: 'red', False: 'blue'},
        hover_data=['speaker', 'text', 'similarity'],
        title="議論の軌跡マップ (Trajectory Map)",
        labels={'is_drift': '脱線候補'}
    )
    # 点に番号を振る
    df_utterances = df[df['speaker'] != 'GOAL']
    for i, row in df_utterances.iterrows():
        fig.add_annotation(x=row['x'], y=row['y'], text=str(i), showarrow=False, yshift=10)

    # 原点(ゴール)を強調
    fig.update_traces(marker=dict(size=12), selector=dict(name='GOAL'))
    fig.add_shape(type="circle", xref="x", yref="y", x0=-0.05, y0=-0.05, x1=0.05, y1=0.05, line_color="blue")


    return fig

# --- サイドバー ---
st.sidebar.title("コントロールパネル")
dev_mode = st.sidebar.checkbox("UI開発モード (API通信なし)", value=False)
st.sidebar.markdown("---")
selected_file = st.sidebar.selectbox(
    "1. 分析データを選択",
    glob.glob("data/mock_discussion_*.json"),
    format_func=lambda x: x.split('/')[-1], # ファイル名だけ表示
    disabled=dev_mode
)
similarity_threshold = st.sidebar.slider("2. 類似度の閾値", 0.0, 1.0, 0.35, 0.01, disabled=dev_mode)
run_analysis = st.sidebar.button("分析を実行", disabled=dev_mode)

# --- メインエリア ---
if dev_mode:
    # (UI開発モードはベクトルデータがないため2Dプロットは一旦非表示)
    st.info("UI開発モードが有効です。2Dプロットは表示されません。")
    st.session_state.analysis_results = load_json_data("data/mock_analysis_result_01.json")
    st.session_state.goal_embedding = None
    st.session_state.last_run_params = {"file": "data/mock_discussion_01.json (ダミー)", "threshold": 0.25}

elif run_analysis:
    if selected_file:
        with st.spinner("分析を実行中... (API通信中)"):
            discussion_data = load_json_data(selected_file)
            detector = DriftDetector(similarity_threshold=similarity_threshold)
            results, goal_embedding = detector.analyze(discussion_data)
            st.session_state.analysis_results = results
            st.session_state.goal_embedding = goal_embedding
        st.session_state.last_run_params = {"file": selected_file, "threshold": similarity_threshold}
    else:
        st.warning("分析するデータファイルが見つかりません。")

# --- 結果表示 ---
if 'analysis_results' in st.session_state and st.session_state.analysis_results:
    last_params = st.session_state.last_run_params
    st.subheader("分析結果")
    st.write(f"**分析対象:** `{last_params['file']}` | **類似度閾値:** `{last_params['threshold']}`")
    
    goal_file = "data/mock_discussion_01.json" if dev_mode else last_params['file']
    discussion_data = load_json_data(goal_file)
    st.info(f"**議論のゴール:** {discussion_data.get('goal', 'N/A')}")
    
    # 2Dプロット (UI開発モードでは表示しない)
    if not dev_mode and 'goal_embedding' in st.session_state and st.session_state.goal_embedding:
        st.subheader("議論の軌跡マップ (可視化 Lv.2)")
        plot_fig = create_2d_plot(st.session_state.analysis_results, st.session_state.goal_embedding, last_params['threshold'])
        st.plotly_chart(plot_fig, use_container_width=True)

    with st.expander("詳細な分析結果を表示 (リスト形式)"):
        for result in st.session_state.analysis_results:
            is_drift_candidate = result['similarity'] < last_params['threshold']
            display_text = f"**[{result['speaker']}]** {result['text']}"
            details = f"類似度: {result['similarity']:.4f}"
            if result.get('llm_judgment'):
                details += f" | LLMの判定: **{result['llm_judgment']}**"
            
            if is_drift_candidate:
                st.error(display_text)
            else:
                st.info(display_text)
            
            st.caption(details)
            st.markdown("---")
else:
    if not dev_mode:
        st.info("サイドバーでパラメータを設定し、「分析を実行」ボタンを押してください。")