# -*- coding: utf-8 -*-
import re
import io
import os
from datetime import datetime
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# 日本語トークナイザ（軽量）：fugashi + unidic-lite
try:
    from fugashi import Tagger
    _ja = Tagger()
    HAS_JA = True
except Exception:
    HAS_JA = False

TITLE = "Instagram キャプション分析"

# ========= ユーティリティ =========
URL_RE       = re.compile(r"https?://\S+")
MENTION_RE   = re.compile(r"(?<!\w)@\w+")
HASHTAG_ALL  = re.compile(r"#([^\s#]+)")  # 日本語ハッシュタグも拾う
EMOJI_RE     = re.compile(r"[\U00010000-\U0010FFFF]", flags=re.UNICODE)

EN_WORD_RE   = re.compile(r"[A-Za-z]+")

EN_STOP = set("""
the of and to in a for on is are was were be been being with as by from that this it an or at
we you they he she our your their my me i us them his her its into about over out up down not no
""".split())

# 品詞を絞る（名詞・固有名詞・形容詞・動詞）
JA_POS_KEEP = {"名詞", "固有名詞", "形容詞", "動詞"}

# 日本語用の簡易ストップワード（必要に応じて拡張してください）
JA_STOP = {
    "こと", "もの", "よう", "さん", "ます", "する", "いる", "なる", "これ", "それ", "ため",
    "ところ", "から", "まで", "など", "こと", "その", "これ", "あの", "どの", "私", "僕", "あなた",
}

# さらに除去したい一般動詞など（必要に応じて拡張）
JA_STOP_VERBS = {"ある", "有る", "為る", "いる", "する", "なる", "できる", "やる", "行く", "来る"}

# トークン長とストップワードのデフォルト設定（UI で上書き可）
MIN_TOKEN_LEN = 2
FILTER_STOPWORDS = True
# トークン抽出モード: 'mixed' (名詞+形容詞+動詞), 'nouns' (名詞のみ), 'proper' (固有名詞のみ)
TOKEN_EXTRACTION_MODE = "proper"

def extract_hashtags(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [m.strip() for m in HASHTAG_ALL.findall(text)]

def clean_caption_for_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = URL_RE.sub(" ", text)
    s = MENTION_RE.sub(" ", s)
    # ハッシュタグ本体はタグ分析用に別抽出するので本文側からは除去
    s = HASHTAG_ALL.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_mixed(text: str) -> list[str]:
    """
    日本語：fugashiがあれば形態素、なければ簡易分割
    英語：単純に英字連続を単語に
    """
    toks = []
    # 入力が文字列でない場合は安全に空配列を返す
    if not isinstance(text, str):
        return toks

    if HAS_JA:
        for w in _ja(text):
            # fugashi の feature が None のケースや属性が欠けているケースに備える
            feat = getattr(w, "feature", None)
            if not feat:
                continue

            # part_of_speech の候補属性名を試す
            pos_raw = None
            for attr in ("part_of_speech", "pos", "pos1", "pos_class"):
                pos_raw = getattr(feat, attr, None)
                if pos_raw:
                    break
            if not pos_raw:
                # feature がカンマ区切りの文字列だった場合のフォールバック
                try:
                    pos_raw = str(feat).split(",")[0]
                except Exception:
                    pos_raw = ""
            pos_parts = (pos_raw or "").split(",")
            pos_main = pos_parts[0] if len(pos_parts) > 0 else ""
            pos_sub = pos_parts[1] if len(pos_parts) > 1 else ""

            # 抽出モードに応じたフィルタリング
            keep_token = False
            if TOKEN_EXTRACTION_MODE == "proper":
                # 固有名詞のみ: pos 主が 名詞 かつ sub が 固有名詞、または主が  固有名詞
                if pos_main == "名詞" and pos_sub == "固有名詞":
                    keep_token = True
                if pos_main == "固有名詞":
                    keep_token = True
            elif TOKEN_EXTRACTION_MODE == "nouns":
                if pos_main == "名詞":
                    keep_token = True
            else:
                # mixed (既存の挙動に近い)
                if pos_main in JA_POS_KEEP:
                    keep_token = True

            if not keep_token:
                continue

            # lemma の候補属性名を試す
            lemma_raw = None
            for attr in ("lemma", "base_form", "dictionary_form", "normalized_form"):
                lemma_raw = getattr(feat, attr, None)
                if lemma_raw:
                    break
            if not lemma_raw:
                lemma_raw = getattr(w, "surface", "")
            lemma = str(lemma_raw).strip()
            # 英字を含む場合のみ小文字化（日本語はそのまま）
            if re.search(r"[A-Za-z]", lemma):
                lemma = lemma.lower()
            if lemma:
                toks.append(lemma)
    else:
        # 日本語分かち書きなしの簡易：記号除去してスペース分割
        s = re.sub(r"[^\w\s]", " ", text)
        for t in s.split():
            t = t.lower().strip()
            if t:
                toks.append(t)

    # 英語トークン追加（日本語の中に英語がある場合）
    for m in EN_WORD_RE.finditer(text):
        w = m.group(0).lower()
        if w and w not in EN_STOP:
            toks.append(w)

    # 後処理フィルタ: ストップワード、数字のみ、短いトークンを除去
    filtered = []
    for t in toks:
        # ストップワード除去
        if FILTER_STOPWORDS:
            if t in EN_STOP or t in JA_STOP or t in JA_STOP_VERBS:
                continue
        # 数字だけのトークンを除去
        if re.fullmatch(r"\d+", t):
            continue
        # 文字数によるフィルタ（日本語は文字数で判断）
        if len(t) < MIN_TOKEN_LEN:
            continue
        filtered.append(t)

    return filtered

def build_counts(list_of_lists):
    c = Counter()
    for arr in list_of_lists:
        c.update(arr)
    return c

def build_cooccurrence(
    df_tokens,
    df_tags,
    mode="all",
    min_count_nodes=2,
    min_weight=2,
    *,
    top_n_words=None,
    top_n_tags=None,
    max_tokens_per_doc=None,
    max_edges_per_node=None,
):
    """
    mode:
      - "words": 本文キーワード同士
      - "tags" : ハッシュタグ同士
      - "cross": キーワード × ハッシュタグ のみ
      - "all"  : すべて（words + tags + cross）
    """
    G = nx.Graph()

    # グローバル頻度で上位のみを残す場合の準備
    token_counter = Counter()
    tag_counter = Counter()
    if top_n_words is not None:
        for toks in df_tokens:
            token_counter.update(toks)
        allowed_tokens = {t for t, _ in token_counter.most_common(int(top_n_words))}
    else:
        allowed_tokens = None

    if top_n_tags is not None:
        for tags in df_tags:
            tag_counter.update(tags)
        allowed_tags = {t for t, _ in tag_counter.most_common(int(top_n_tags))}
    else:
        allowed_tags = None

    for toks, tags in zip(df_tokens, df_tags):
        # 文書中のトークン/タグをセット化し、global filter を通す
        toks_list = [t for t in toks if (allowed_tokens is None or t in allowed_tokens)]
        tags_list = [t for t in tags if (allowed_tags is None or t in allowed_tags)]

        # 文書内のトークン数を制限する（高頻度トークン優先）
        if max_tokens_per_doc is not None and len(toks_list) > max_tokens_per_doc:
            # トークンの優先度をグローバル頻度でソート
            if token_counter:
                toks_list = sorted(toks_list, key=lambda x: token_counter.get(x, 0), reverse=True)[:max_tokens_per_doc]
            else:
                toks_list = toks_list[:max_tokens_per_doc]

        toks_set = set(toks_list)
        tags_set = set(tags_list)

        # nodes を登録（頻度フィルタは後段で）
        for t in toks_set:
            G.add_node(t, ntype="word")
        for h in tags_set:
            G.add_node("#"+h, ntype="tag")

        # エッジ追加
        if mode in ("all", "words"):
            # 本文語 × 本文語
            arr = sorted(toks_set)
            for i in range(len(arr)):
                for j in range(i+1, len(arr)):
                    u, v = arr[i], arr[j]
                    G.add_edge(u, v, weight=G.get_edge_data(u, v, {}).get("weight", 0)+1)

        if mode in ("all", "tags"):
            # タグ × タグ
            arr = sorted(tags_set)
            for i in range(len(arr)):
                for j in range(i+1, len(arr)):
                    u, v = "#"+arr[i], "#"+arr[j]
                    G.add_edge(u, v, weight=G.get_edge_data(u, v, {}).get("weight", 0)+1)

        if mode in ("all", "cross"):
            # 本文語 × タグ
            for t in toks_set:
                for h in tags_set:
                    u, v = t, "#"+h
                    G.add_edge(u, v, weight=G.get_edge_data(u, v, {}).get("weight", 0)+1)

    # ノード出現頻度（次数ベース）で閾値
    # 次数 + 自己出現の近似として、隣接エッジ重みの合計を使用
    node_strength = defaultdict(int)
    for u, v, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        node_strength[u] += w
        node_strength[v] += w

    nodes_to_keep = {n for n, s in node_strength.items() if s >= min_count_nodes}
    G = G.subgraph(nodes_to_keep).copy()

    # エッジ重みフィルタ
    to_drop = []
    for u, v, d in G.edges(data=True):
        if int(d.get("weight", 1)) < min_weight:
            to_drop.append((u, v))
    G.remove_edges_from(to_drop)

    # 孤立ノード除去
    iso = list(nx.isolates(G))
    G.remove_nodes_from(iso)

    # 各ノードについてエッジ数を制限する（重みの低いエッジを切る）
    if max_edges_per_node is not None:
        to_remove = set()
        for u in list(G.nodes()):
            nbrs = []
            for v, attrs in G[u].items():
                w = int(attrs.get("weight", 1))
                nbrs.append((v, w))
            if len(nbrs) <= max_edges_per_node:
                continue
            nbrs_sorted = sorted(nbrs, key=lambda x: x[1], reverse=True)
            keep = {v for v, _ in nbrs_sorted[:max_edges_per_node]}
            for v, _ in nbrs:
                if v not in keep:
                    to_remove.add(tuple(sorted((u, v))))
        # 削除
        for u, v in to_remove:
            if G.has_edge(u, v):
                G.remove_edge(u, v)

    return G, node_strength

def pyvis_html(G, node_strength, height="700px"):
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="black", notebook=False, directed=False)
    # レイアウトを少し落ち着かせる
    net.barnes_hut(gravity=-8000, central_gravity=0.2, spring_length=120, spring_strength=0.01)

    for n, data in G.nodes(data=True):
        ntype = data.get("ntype", "word")
        size = max(8, min(60, node_strength.get(n, 1)))  # 簡易スケーリング
        title = f"{n} ({'word' if ntype=='word' else 'tag'})"
        color = "#4C78A8" if ntype == "word" else "#F58518"
        net.add_node(n, label=n, title=title, size=size, color=color, shape="dot")

    for u, v, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(u, v, value=w, title=f"co-occurrence: {w}")

    tmp_html = "network_preview.html"
    # notebook モードでのテンプレートレンダリングが環境依存で失敗する場合があるため
    # ここでは notebook=False で HTML を直接書き出す（net.show は notebook=True を使う）
    net.write_html(tmp_html, open_browser=False, notebook=False)
    with open(tmp_html, "r", encoding="utf-8") as f:
        html = f.read()
    return html

# ========= UI =========
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

st.sidebar.header("データ入力")
uploaded = st.sidebar.file_uploader("CSVをアップロード（caption列 必須）", type=["csv"])
default_path = st.sidebar.text_input("またはファイルパス指定", value="")
# サンプルデータで手早く試すボタン
use_demo = st.sidebar.button("デモデータで試す")

# フィルタ設定
st.sidebar.header("フィルタ・解析設定")
date_enable = st.sidebar.checkbox("timestamp（日付）で絞り込む", value=False)
date_col = st.sidebar.text_input("timestamp 列名", value="timestamp")
mode = st.sidebar.selectbox("共起モード", ["all", "words", "tags", "cross"], index=0)
min_node = st.sidebar.number_input("ノードの最小出現（強度）", min_value=1, max_value=999, value=2, step=1)
min_edge = st.sidebar.number_input("エッジ最小重み", min_value=1, max_value=999, value=2, step=1)
topn_words = st.sidebar.slider("頻出キーワードの表示数", 10, 200, 50, 10)
topn_tags  = st.sidebar.slider("頻出ハッシュタグの表示数", 10, 200, 50, 10)

# トークン表示の微調整（短すぎる語やストップワードを除去）
min_token_len = st.sidebar.slider("最小トークン長（文字数）", 1, 5, MIN_TOKEN_LEN, 1)
filter_stopwords = st.sidebar.checkbox("ストップワード除去（日本語/英語）", value=FILTER_STOPWORDS)

# 抽出モード: 固有名詞のみ/名詞のみ/デフォルト(mixed)
mode_label = st.sidebar.selectbox("トークン抽出モード", ["固有名詞のみ", "名詞のみ", "名詞/形容詞/動詞(mixed)"], index=1)
if mode_label == "固有名詞のみ":
    TOKEN_EXTRACTION_MODE = "proper"
elif mode_label == "名詞のみ":
    TOKEN_EXTRACTION_MODE = "nouns"
else:
    TOKEN_EXTRACTION_MODE = "mixed"

st.sidebar.markdown("---")
st.sidebar.caption("注: 日本語分かち書きは fugashi を利用。未導入でも英語/簡易トークンで動きます。")

# ========= ロード =========
df = None
src = ""
if use_demo:
    try:
        df = pd.read_csv("sample.csv")
        src = "sample.csv (demo)"
    except Exception as e:
        st.error(f"デモデータ読み込みエラー: {e}")
        df = None
        src = ""
if uploaded is not None:
    df = pd.read_csv(uploaded)
    src = uploaded.name
elif default_path:
    try:
        df = pd.read_csv(default_path)
        src = default_path
    except Exception as e:
        st.error(f"CSV読込エラー: {e}")

if df is None:
    st.info("左のサイドバーから CSV をアップロードするか、ファイルパスを指定してください。")
    st.stop()

st.write(f"**読み込み元**: `{src}`  |  行数: {len(df)}")

if "caption" not in df.columns:
    st.error("このCSVに 'caption' 列がありません。列名を確認してください。")
    st.stop()

# ========= 日付フィルタ =========
if date_enable:
    if date_col not in df.columns:
        st.warning(f"指定の timestamp 列 '{date_col}' が見つかりません。フィルタをスキップします。")
    else:
        # datetimeに変換
        try:
            df["_dt_"] = pd.to_datetime(df[date_col], errors="coerce")
            min_dt = pd.to_datetime(df["_dt_"].min())
            max_dt = pd.to_datetime(df["_dt_"].max())
            r = st.slider("期間を選択", value=(min_dt, max_dt), min_value=min_dt, max_value=max_dt)
            m1 = (df["_dt_"] >= r[0]) & (df["_dt_"] <= r[1])
            df = df.loc[m1].copy()
            st.write(f"期間で絞り込み後の行数: {len(df)}")
        except Exception as e:
            st.warning(f"timestamp 解析に失敗: {e}")

# ========= 抽出・前処理 =========
with st.spinner("前処理中..."):
    df["__hashtags__"] = df["caption"].apply(extract_hashtags)
    df["__text__"] = df["caption"].apply(clean_caption_for_text)
    # サイドバーで設定したトークンフィルタをモジュール変数に適用
    MIN_TOKEN_LEN = int(min_token_len)
    FILTER_STOPWORDS = bool(filter_stopwords)
    df["__tokens__"] = df["__text__"].apply(tokenize_mixed)

# ========= 頻度集計 =========
word_counts = build_counts(df["__tokens__"].tolist())
tag_counts  = build_counts(df["__hashtags__"].tolist())

# 表示
st.subheader("頻出キーワード（本文）")
if len(word_counts)==0:
    st.write("（本文トークンがありません）")
else:
    wc_df = pd.DataFrame(word_counts.most_common(topn_words), columns=["token", "count"])
    st.dataframe(wc_df, use_container_width=True)
    csv = wc_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ キーワード頻度CSVをダウンロード", data=csv, file_name="word_freq.csv", mime="text/csv")

st.subheader("頻出ハッシュタグ")
if len(tag_counts)==0:
    st.write("（ハッシュタグがありません）")
else:
    tc_df = pd.DataFrame(tag_counts.most_common(topn_tags), columns=["hashtag", "count"])
    st.dataframe(tc_df, use_container_width=True)
    csv2 = tc_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ ハッシュタグ頻度CSVをダウンロード", data=csv2, file_name="hashtag_freq.csv", mime="text/csv")

# ========= 共起ネットワーク =========
st.subheader("共起ネットワーク")
with st.spinner("共起ネットワークを生成中..."):
    G, node_strength = build_cooccurrence(
        df["__tokens__"].tolist(),
        df["__hashtags__"].tolist(),
        mode=mode,
        min_count_nodes=int(min_node),
        min_weight=int(min_edge),
        # UI の topn を利用してノード数を抑える
        top_n_words=int(topn_words),
        top_n_tags=int(topn_tags),
        # 文書ごとのトークン数を制限（軽量化）
        max_tokens_per_doc=10,
        # 各ノードが持つエッジ数を制限（さらに軽量化）
        max_edges_per_node=10,
    )

    if G.number_of_nodes() == 0:
        st.write("（条件に合致する共起ネットワークがありません）")
    else:
        html = pyvis_html(G, node_strength, height="750px")
        components.html(html, height=780, scrolling=True)

        # HTMLエクスポート
        st.download_button(
            "⬇️ ネットワークHTMLをダウンロード",
            data=html.encode("utf-8"),
            file_name="cooccurrence_network.html",
            mime="text/html"
        )

# ========= メモ =========
#with st.expander("注意点・調整のヒント", expanded=False):
#    st.markdown("""
#- fugashi（日本語形態素解析）が入っていない場合は簡易トークンで動きます。`pip install fugashi[unidic-lite]` 推奨  
#- うまく分割されない専門語は、辞書登録 or 正規表現ルールの追加で対応可能  
#- 共起の「最小出現（ノード強度）」と「最小重み（エッジ）」でグラフの密度を調整できます  
#- 本文語/タグ/クロスのどれを見るかは *共起モード* で切り替え  
#- もっと本格的にやるなら TF-IDF、時系列（グルーピング）等も容易に追加可能
#""")