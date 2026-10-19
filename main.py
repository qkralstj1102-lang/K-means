
import math

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==================================================
# 1. 페이지 설정
# ==================================================
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)

FEATURES = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "longrun_index",
}

CLUSTER_SYMBOLS = [
    "㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"
]


# ==================================================
# 2. 제목
# ==================================================
st.title("🎬 영화 유형 나누기")
st.caption(
    "영화 데이터를 분석하고 K-평균 알고리즘으로 "
    "비슷한 특성을 가진 영화들을 유형별로 분류합니다."
)

st.divider()


# ==================================================
# 3. 데이터 불러오기 및 전처리
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd",
        "movieNm",
        "openDt",
        "genre",
        "nation",
        "first_scrn",
        "first_show",
        "first_date",
        "peak",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    df = df.copy()

    numeric_columns = [
        "first_scrn",
        "first_show",
        "peak",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        )

    # 제목이 없는 행 제외
    df = df.dropna(subset=["movieNm"]).copy()

    total_count = len(df)

    # 스크린 수와 누적 관객에 상용로그 적용
    df["log_first_scrn"] = df["first_scrn"].apply(
        lambda x: math.log10(x)
        if pd.notna(x) and x > 0
        else float("nan")
    )

    df["log_total_audi"] = df["total_audi"].apply(
        lambda x: math.log10(x)
        if pd.notna(x) and x > 0
        else float("nan")
    )

    # 롱런 지수 = 누적 관객 / 첫 주 관객
    df["longrun_index"] = float("nan")

    valid_first_week = (
        df["first_week_audi"].notna()
        & (df["first_week_audi"] > 0)
    )

    df.loc[valid_first_week, "longrun_index"] = (
        df.loc[valid_first_week, "total_audi"]
        / df.loc[valid_first_week, "first_week_audi"]
    )

    # 롱런 지수 최대 20
    df["longrun_index"] = df["longrun_index"].clip(upper=20)

    feature_columns = [
        "log_first_scrn",
        "log_total_audi",
        "days_in_top10",
        "longrun_index",
    ]

    valid_mask = (
        df[feature_columns].notna().all(axis=1)
        & (df["first_week_audi"] > 0)
        & (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
    )

    clean_df = df.loc[valid_mask].copy().reset_index(drop=True)

    return clean_df, total_count


try:
    df, total_count = load_data()
except Exception as e:
    st.error("데이터를 불러오거나 처리하는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


st.write(
    f"**전체 영화: {total_count:,}편**"
    f"　|　**분석에 사용한 영화: {len(df):,}편**"
)

st.caption(
    "필요한 속성에 결측치가 있거나 첫 주 관객이 0인 영화, "
    "로그 계산이 불가능한 영화는 분석에서 제외합니다."
)

if len(df) < 3:
    st.error("분석 가능한 영화가 최소 3편 필요합니다.")
    st.stop()

st.divider()


# ==================================================
# 4. 군집화 속성 및 군집 수 선택
# ==================================================
st.subheader("1. 군집화 설정")

selected_features = st.multiselect(
    "군집화에 사용할 속성",
    options=list(FEATURES.keys()),
    default=list(FEATURES.keys()),
    help="두 개 이상 선택하세요. 기본값은 네 가지 속성 전체입니다.",
)

max_clusters = min(7, len(df))

selected_k = st.selectbox(
    "군집 수",
    options=list(range(2, max_clusters + 1)),
    index=min(3, max_clusters - 2),
    help="2개부터 7개까지 선택할 수 있습니다. 기본값은 3개입니다.",
)

if len(selected_features) < 2:
    st.warning("군집화 속성을 두 개 이상 선택해 주세요.")
    st.stop()

feature_columns = [
    FEATURES[name] for name in selected_features
]


# ==================================================
# 5. 선택한 속성 표준화
# ==================================================
X = df[feature_columns].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==================================================
# 6. WCSS 계산: 군집 수 1~7
# ==================================================
st.divider()
st.subheader("2. 적절한 군집 수 찾기 - 엘보 그래프")

wcss_values = []

for k in range(1, max_clusters + 1):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    model.fit(X_scaled)

    # inertia_ = 군집 중심과 각 데이터 점 사이의
    # 제곱 거리 합(WCSS)
    wcss_values.append({
        "군집 수": k,
        "WCSS": float(model.inertia_),
    })

wcss_df = pd.DataFrame(wcss_values)

fig_wcss = go.Figure()

fig_wcss.add_trace(
    go.Scatter(
        x=wcss_df["군집 수"],
        y=wcss_df["WCSS"],
        mode="lines+markers",
        name="WCSS",
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "군집 수: %{x}"
            "<br>WCSS: %{y:,.3f}"
            "<extra></extra>"
        ),
    )
)

# 현재 선택한 군집 수에 세로선 표시
fig_wcss.add_vline(
    x=selected_k,
    line_dash="dash",
    line_width=2,
    annotation_text=f"현재 선택: {selected_k}개",
    annotation_position="top",
)

fig_wcss.update_layout(
    title="군집 수에 따른 WCSS 변화",
    xaxis_title="군집 수",
    yaxis_title="군집 내 제곱 거리 합 (WCSS)",
    xaxis=dict(
        tickmode="linear",
        dtick=1,
    ),
    height=470,
    margin=dict(l=20, r=20, t=80, b=30),
    hovermode="x unified",
)

st.plotly_chart(fig_wcss, use_container_width=True)

st.markdown(
    """
    **그래프 해석 방법**

    - WCSS는 같은 군집에 속한 점들이 각 군집 중심에서 얼마나 떨어져 있는지를 나타냅니다.
    - 군집 수가 증가하면 일반적으로 WCSS는 감소합니다.
    - 그래프의 감소 폭이 크게 줄어드는 지점은 군집 수를 결정할 때 참고할 수 있습니다.
    - WCSS가 가장 작은 군집 수가 반드시 가장 좋은 선택이라는 뜻은 아닙니다.
    """
)

# WCSS 및 직전 군집 수와의 차이
wcss_table = wcss_df.copy()
wcss_table["직전 값 대비 감소량"] = (
    wcss_table["WCSS"].shift(1) - wcss_table["WCSS"]
)

# 첫 행은 비교 대상이 없으므로 빈칸
wcss_table["직전 값 대비 감소량"] = (
    wcss_table["직전 값 대비 감소량"].round(3)
)
wcss_table.loc[0, "직전 값 대비 감소량"] = float("nan")

wcss_table["WCSS"] = wcss_table["WCSS"].round(3)

st.dataframe(
    wcss_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "군집 수": st.column_config.NumberColumn(
            "군집 수",
            format="%d개",
        ),
        "WCSS": st.column_config.NumberColumn(
            "WCSS",
            format="%,.3f",
        ),
        "직전 값 대비 감소량": st.column_config.NumberColumn(
            "직전 값 대비 감소량",
            format="%,.3f",
        ),
    },
)

st.divider()


# ==================================================
# 7. 선택한 군집 수로 K-평균 실행
# ==================================================
kmeans = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=10,
)

df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 높은 군집부터 표시 기호 지정
cluster_means = (
    df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
)

cluster_order = cluster_means.index.tolist()

label_map = {
    cluster_id: CLUSTER_SYMBOLS[index]
    for index, cluster_id in enumerate(cluster_order)
}

df["유형"] = df["cluster_raw"].map(label_map)

active_labels = CLUSTER_SYMBOLS[:selected_k]


# ==================================================
# 8. 실루엣 점수
# ==================================================
st.subheader("3. 현재 군집화의 실루엣 점수")

try:
    if len(set(df["cluster_raw"])) < 2:
        silhouette = float("nan")
    elif len(set(df["cluster_raw"])) >= len(df):
        silhouette = float("nan")
    else:
        silhouette = silhouette_score(
            X_scaled,
            df["cluster_raw"],
        )

    if pd.notna(silhouette):
        st.write(
            f"**선택한 군집 수: {selected_k}개**"
            f"　|　**실루엣 점수: {silhouette:.4f}**"
        )

        st.progress(
            max(0.0, min(1.0, (silhouette + 1) / 2))
        )

        st.caption(
            "실루엣 점수는 -1부터 1 사이입니다. "
            "1에 가까울수록 각 영화가 자기 군집에 잘 속하고 "
            "다른 군집과 잘 구분되는 경향이 있습니다. "
            "0에 가까우면 군집이 겹칠 수 있으며, "
            "음수이면 일부 영화가 다른 군집에 더 가까울 수 있습니다."
        )
    else:
        st.warning(
            "현재 데이터로 실루엣 점수를 계산할 수 없습니다."
        )

except Exception as e:
    st.warning(f"실루엣 점수 계산 중 문제가 발생했습니다: {e}")

st.info(
    f"현재 **{selected_k}개 군집**으로 분석하고 있습니다. "
    "유형 이름은 누적 관객 평균이 높은 순서로 표시합니다."
)

st.divider()


# ==================================================
# 9. 2차원 산점도
# ==================================================
st.subheader("4. 2차원 산점도")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        key="scatter_2d_x",
    )

with col_y:
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=1,
        key="scatter_2d_y",
    )

fig_2d = px.scatter(
    df,
    x=FEATURES[x_feature],
    y=FEATURES[y_feature],
    color="유형",
    category_orders={"유형": active_labels},
    hover_name="movieNm",
    hover_data={
        "유형": True,
        FEATURES[x_feature]: ":.3f",
        FEATURES[y_feature]: ":.3f",
        "movieNm": False,
    },
    labels={
        FEATURES[x_feature]: x_feature,
        FEATURES[y_feature]: y_feature,
        "유형": "영화 유형",
    },
    title=f"{x_feature} × {y_feature}",
)

fig_2d.update_traces(
    marker=dict(size=8, opacity=0.8)
)

fig_2d.update_layout(
    height=550,
    legend_title_text="영화 유형",
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig_2d, use_container_width=True)

st.caption(
    "점 하나가 영화 한 편입니다. 마우스를 올리면 영화 제목과 "
    "선택한 두 속성의 값을 확인할 수 있습니다."
)

st.divider()


# ==================================================
# 10. 3차원 산점도
# ==================================================
st.subheader("5. 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 군집화 속성을 "
        "세 개 이상 선택해 주세요."
    )
else:
    col_x3, col_y3, col_z3 = st.columns(3)

    with col_x3:
        x3 = st.selectbox(
            "X축 속성",
            options=selected_features,
            index=0,
            key="scatter_3d_x",
        )

    with col_y3:
        y3 = st.selectbox(
            "Y축 속성",
            options=selected_features,
            index=1,
            key="scatter_3d_y",
        )

    with col_z3:
        z3 = st.selectbox(
            "Z축 속성",
            options=selected_features,
            index=2,
            key="scatter_3d_z",
        )

    fig_3d = px.scatter_3d(
        df,
        x=FEATURES[x3],
        y=FEATURES[y3],
        z=FEATURES[z3],
        color="유형",
        category_orders={"유형": active_labels},
        hover_name="movieNm",
        hover_data={
            "유형": True,
            FEATURES[x3]: ":.3f",
            FEATURES[y3]: ":.3f",
            FEATURES[z3]: ":.3f",
            "movieNm": False,
        },
        labels={
            FEATURES[x3]: x3,
            FEATURES[y3]: y3,
            FEATURES[z3]: z3,
            "유형": "영화 유형",
        },
        title="영화 유형의 3차원 분포",
    )

    fig_3d.update_traces(
        marker=dict(size=3, opacity=0.8)
    )

    fig_3d.update_layout(
        height=650,
        legend_title_text="영화 유형",
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title=x3,
            yaxis_title=y3,
            zaxis_title=z3,
        ),
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    st.caption(
        "마우스로 드래그해 그래프를 회전할 수 있습니다. "
        "점에 마우스를 올리면 영화 제목을 확인할 수 있습니다."
    )

st.divider()


# ==================================================
# 11. 유형별 통계
# ==================================================
st.subheader("6. 유형별 통계")

stats = (
    df.groupby("유형")
    .agg(
        movie_count=("movieNm", "count"),
        avg_first_scrn=("first_scrn", "mean"),
        avg_total_audi=("total_audi", "mean"),
        avg_days_in_top10=("days_in_top10", "mean"),
        avg_longrun_index=("longrun_index", "mean"),
    )
    .reindex(active_labels)
)

stats_display = stats.rename(
    columns={
        "movie_count": "영화 편수",
        "avg_first_scrn": "평균 스크린 수",
        "avg_total_audi": "평균 누적 관객",
        "avg_days_in_top10": "평균 10위권 일수",
        "avg_longrun_index": "평균 롱런 지수",
    }
).copy()

stats_display["영화 편수"] = (
    stats_display["영화 편수"].fillna(0).astype(int)
)

stats_display["평균 스크린 수"] = (
    stats_display["평균 스크린 수"].round(1)
)

stats_display["평균 누적 관객"] = (
    stats_display["평균 누적 관객"].round(1)
)

stats_display["평균 10위권 일수"] = (
    stats_display["평균 10위권 일수"].round(2)
)

stats_display["평균 롱런 지수"] = (
    stats_display["평균 롱런 지수"].round(2)
)

st.dataframe(
    stats_display,
    use_container_width=True,
)

st.caption(
    "스크린 수와 누적 관객 평균은 로그 변환 전의 원래 단위입니다. "
    "군집 수를 바꾸면 이 표도 다시 계산됩니다."
)

st.divider()


# ==================================================
# 12. 유형별 누적 관객 TOP 5
# ==================================================
st.subheader("7. 유형별 대표 영화 TOP 5")

for label in active_labels:
    group_df = df[df["유형"] == label].copy()

    group_df = group_df.sort_values(
        "total_audi",
        ascending=False,
    ).head(5)

    avg_audience = stats.loc[label, "avg_total_audi"]
    movie_count = int(stats.loc[label, "movie_count"])

    if pd.isna(avg_audience):
        avg_audience_text = "정보 없음"
    else:
        avg_audience_text = f"{avg_audience:,.0f}명"

    with st.expander(
        f"{label} 유형 | {movie_count:,}편 | "
        f"평균 누적 관객 {avg_audience_text}",
        expanded=True,
    ):
        if group_df.empty:
            st.write("이 유형에 속한 영화가 없습니다.")
        else:
            top_movies = group_df[
                [
                    "movieNm",
                    "total_audi",
                    "first_week_audi",
                    "days_in_top10",
                    "longrun_index",
                ]
            ].copy()

            top_movies.columns = [
                "영화 제목",
                "누적 관객",
                "첫 주 관객",
                "10위권 일수",
                "롱런 지수",
            ]

            top_movies["누적 관객"] = (
                top_movies["누적 관객"]
                .map(lambda x: f"{x:,.0f}")
            )

            top_movies["첫 주 관객"] = (
                top_movies["첫 주 관객"]
                .map(lambda x: f"{x:,.0f}")
            )

            top_movies["10위권 일수"] = (
                top_movies["10위권 일수"]
                .map(
                    lambda x: f"{x:,.0f}"
                    if pd.notna(x) else "-"
                )
            )

            top_movies["롱런 지수"] = (
                top_movies["롱런 지수"]
                .map(
                    lambda x: f"{x:.2f}"
                    if pd.notna(x) else "-"
                )
            )

            top_movies.index = range(1, len(top_movies) + 1)
            top_movies.index.name = "순위"

            st.dataframe(
                top_movies,
                use_container_width=True,
            )

st.divider()


# ==================================================
# 13. 분석 방법
# ==================================================
st.subheader("분석 방법")

st.markdown(
    """
    1. **데이터 전처리:** 필요한 속성에 결측치가 있는 영화를 제외합니다.
    2. **속성 변환:** 스크린 수와 누적 관객에 상용로그를 적용합니다.
    3. **롱런 지수 계산:** 누적 관객을 첫 주 관객으로 나누고 최대 20으로 제한합니다.
    4. **표준화:** 선택한 속성을 표준화해 단위 차이를 줄입니다.
    5. **군집 수 비교:** 1~7개 군집의 WCSS를 계산해 엘보 그래프로 비교합니다.
    6. **군집화:** 선택한 군집 수로 K-평균을 실행합니다.
    7. **군집 평가:** 실루엣 점수로 군집의 분리 정도를 확인합니다.
    8. **유형 표시:** 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰, ㉱, ㉲, ㉳, ㉴ 순으로 표시합니다.
    """
)

st.caption(
    "K-평균은 random_state=42와 n_init=10으로 설정했습니다. "
    "같은 데이터와 같은 속성을 사용하면 결과를 재현할 수 있습니다."
)
