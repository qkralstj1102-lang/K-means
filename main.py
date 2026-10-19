
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------
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
    "롱런 지수": "longrun_index"
}

RAW_FEATURE_NAMES = {
    "스크린 수 (로그)": "스크린 수",
    "누적 관객 (로그)": "누적 관객",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수"
}

FEATURE_COLS = list(FEATURES.keys())
CLUSTER_LABELS = ["㉮", "㉯", "㉰"]

# --------------------------------------------------
# 2. 제목
# --------------------------------------------------
st.title("🎬 영화 유형 나누기")
st.caption(
    "영화 데이터를 바탕으로 비슷한 특성을 가진 영화들을 "
    "K-평균 알고리즘으로 세 가지 유형으로 나눕니다."
)

st.divider()


# --------------------------------------------------
# 3. 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "필요한 열이 없습니다: " + ", ".join(missing_columns)
        )

    df = df.copy()

    # 숫자 열의 쉼표와 공백을 처리
    numeric_columns = [
        "first_scrn",
        "first_show",
        "first_date",
        "peak",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce"
        )

    # 영화 제목이 없는 행은 제외
    df = df.dropna(subset=["movieNm"]).copy()

    # 원래 데이터의 전체 영화 편수
    total_count = len(df)

    # 군집화에 사용할 네 속성을 계산
    # 로그를 취할 수 있도록 스크린 수와 누적 관객이 양수여야 함
    df["log_first_scrn"] = df["first_scrn"].where(
        df["first_scrn"] > 0
    ).apply(lambda x: __import__("math").log10(x) if pd.notna(x) else float("nan"))

    df["log_total_audi"] = df["total_audi"].where(
        df["total_audi"] > 0
    ).apply(lambda x: __import__("math").log10(x) if pd.notna(x) else float("nan"))

    # 첫 주 관객이 0이면 나눗셈이 불가능하므로 제외 대상
    valid_first_week = df["first_week_audi"] > 0

    # 누적 관객 / 첫 주 관객, 최대 20
    df["longrun_index"] = (
        df["total_audi"] / df["first_week_audi"]
    ).clip(upper=20)

    # 네 속성 중 하나라도 없거나 계산할 수 없는 영화는 제외
    valid_mask = (
        df["first_scrn"].notna()
        & df["total_audi"].notna()
        & df["days_in_top10"].notna()
        & df["first_week_audi"].notna()
        & valid_first_week
        & (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
        & df[["log_first_scrn", "log_total_audi",
              "days_in_top10", "longrun_index"]].notna().all(axis=1)
    )

    clean_df = df.loc[valid_mask].copy()

    return clean_df, total_count


try:
    df, total_count = load_data()
except Exception as e:
    st.error("데이터를 불러오거나 처리하지 못했습니다.")
    st.code(str(e))
    st.stop()

st.success(
    f"전체 영화: {total_count:,}편  |  "
    f"분석 가능한 영화: {len(df):,}편"
)

if len(df) < 3:
    st.error("군집화를 하려면 분석 가능한 영화가 최소 3편 필요합니다.")
    st.stop()

st.caption(
    "로그를 적용하는 스크린 수와 누적 관객은 0 이하인 경우 "
    "로그를 계산할 수 없어 분석에서 제외합니다."
)

st.divider()


# --------------------------------------------------
# 4. 군집화에 사용할 속성 선택
# --------------------------------------------------
st.subheader("1. 영화 유형을 나눌 기준 선택")

selected_features = st.multiselect(
    "군집화에 사용할 속성 (2개 이상 선택)",
    options=FEATURE_COLS,
    default=FEATURE_COLS,
    help=(
        "선택한 속성을 표준화한 다음 K-평균 알고리즘으로 "
        "영화를 세 군집으로 나눕니다."
    )
)

if len(selected_features) < 2:
    st.warning("군집화할 속성을 두 개 이상 선택해 주세요.")
    st.stop()

feature_columns = [
    FEATURES[name] for name in selected_features
]

# --------------------------------------------------
# 5. 표준화 및 K-평균 군집화
# --------------------------------------------------
X = df[feature_columns].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 지정
cluster_audience = (
    df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
)

cluster_order = cluster_audience.index.tolist()

label_map = {
    cluster_id: CLUSTER_LABELS[rank]
    for rank, cluster_id in enumerate(cluster_order)
}

df["유형"] = df["cluster_raw"].map(label_map)

st.info(
    "유형 이름은 누적 관객 평균이 높은 군집부터 "
    "㉮, ㉯, ㉰ 순서로 붙였습니다. "
    "이는 군집 번호 자체의 순서와는 다릅니다."
)

st.divider()


# --------------------------------------------------
# 6. 2차원 산점도
# --------------------------------------------------
st.subheader("2. 2차원 산점도")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        key="scatter_2d_x"
    )

with col_y:
    default_y = 1 if len(selected_features) > 1 else 0
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=default_y,
        key="scatter_2d_y"
    )

plot_2d = df.copy()

fig_2d = px.scatter(
    plot_2d,
    x=FEATURES[x_feature],
    y=FEATURES[y_feature],
    color="유형",
    category_orders={"유형": CLUSTER_LABELS},
    hover_name="movieNm",
    hover_data={
        "유형": True,
        FEATURES[x_feature]: ":.3f",
        FEATURES[y_feature]: ":.3f",
        "movieNm": False
    },
    labels={
        FEATURES[x_feature]: x_feature,
        FEATURES[y_feature]: y_feature,
        "유형": "영화 유형"
    },
    title=f"{x_feature} × {y_feature}"
)

fig_2d.update_traces(marker={"size": 9, "opacity": 0.8})
fig_2d.update_layout(
    height=550,
    legend_title_text="영화 유형",
    margin=dict(l=20, r=20, t=65, b=20)
)

st.plotly_chart(fig_2d, use_container_width=True)

st.caption(
    "점 하나가 영화 한 편을 나타냅니다. "
    "점에 마우스를 올리면 영화 제목과 속성 값을 확인할 수 있습니다."
)

st.divider()


# --------------------------------------------------
# 7. 3차원 산점도
# --------------------------------------------------
st.subheader("3. 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 그리려면 군집화 속성을 세 개 이상 선택해 주세요."
    )
else:
    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox(
            "X축 속성",
            options=selected_features,
            index=0,
            key="scatter_3d_x"
        )

    with c2:
        y3 = st.selectbox(
            "Y축 속성",
            options=selected_features,
            index=1,
            key="scatter_3d_y"
        )

    with c3:
        z3 = st.selectbox(
            "Z축 속성",
            options=selected_features,
            index=2,
            key="scatter_3d_z"
        )

    fig_3d = px.scatter_3d(
        df,
        x=FEATURES[x3],
        y=FEATURES[y3],
        z=FEATURES[z3],
        color="유형",
        category_orders={"유형": CLUSTER_LABELS},
        hover_name="movieNm",
        hover_data={
            "유형": True,
            FEATURES[x3]: ":.3f",
            FEATURES[y3]: ":.3f",
            FEATURES[z3]: ":.3f",
            "movieNm": False
        },
        labels={
            FEATURES[x3]: x3,
            FEATURES[y3]: y3,
            FEATURES[z3]: z3,
            "유형": "영화 유형"
        },
        title="영화 유형의 3차원 분포"
    )

    fig_3d.update_traces(
        marker={"size": 3, "opacity": 0.8}
    )

    fig_3d.update_layout(
        height=650,
        legend_title_text="영화 유형",
        margin=dict(l=0, r=0, t=65, b=0),
        scene=dict(
            xaxis_title=x3,
            yaxis_title=y3,
            zaxis_title=z3
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    st.caption(
        "마우스로 드래그하면 시점을 회전할 수 있습니다. "
        "점에 마우스를 올리면 영화 제목을 확인할 수 있습니다."
    )

st.divider()


# --------------------------------------------------
# 8. 군집별 평균 및 편수
# --------------------------------------------------
st.subheader("4. 유형별 통계")

# 로그가 아닌 원래 단위로 평균 계산
stats = (
    df.groupby("유형")
    .agg(
        편수=("movieNm", "count"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        10위권_일수_평균=("days_in_top10", "mean"),
        롱런_지수_평균=("longrun_index", "mean")
    )
    .reindex(CLUSTER_LABELS)
)

stats_display = stats.copy()
stats_display.columns = [
    "영화 편수",
    "평균 스크린 수",
    "평균 누적 관객",
    "평균 10위권 일수",
    "평균 롱런 지수"
]

stats_display["영화 편수"] = stats_display["영화 편수"].astype(int)
stats_display["평균 스크린 수"] = stats_display["평균 스크린 수"].round(1)
stats_display["평균 누적 관객"] = stats_display["평균 누적 관객"].round(1)
stats_display["평균 10위권 일수"] = stats_display["평균 10위권 일수"].round(2)
stats_display["평균 롱런 지수"] = stats_display["평균 롱런 지수"].round(2)

st.dataframe(
    stats_display,
    use_container_width=True
)

st.caption(
    "평균은 로그를 취하기 전의 원래 스크린 수와 누적 관객을 사용합니다. "
    "롱런 지수는 누적 관객을 첫 주 관객으로 나눈 뒤 20으로 제한한 값입니다."
)

st.divider()


# --------------------------------------------------
# 9. 군집별 누적 관객 상위 5편
# --------------------------------------------------
st.subheader("5. 유형별 대표 영화 TOP 5")

for label in CLUSTER_LABELS:
    group_df = df[df["유형"] == label].copy()
    group_df = group_df.sort_values(
        "total_audi",
        ascending=False
    ).head(5)

    avg_audience = stats.loc[label, "누적_관객_평균"]
    movie_count = int(stats.loc[label, "편수"])

    with st.expander(
        f"{label} 유형 — 영화 {movie_count:,}편 · "
        f"평균 누적 관객 {avg_audience:,.0f}명",
        expanded=True
    ):
        if group_df.empty:
            st.write("해당 유형에 속한 영화가 없습니다.")
        else:
            top_movies = group_df[
                ["movieNm", "total_audi", "first_week_audi",
                 "days_in_top10", "longrun_index"]
            ].copy()

            top_movies.columns = [
                "영화 제목",
                "누적 관객",
                "첫 주 관객",
                "10위권 일수",
                "롱런 지수"
            ]

            top_movies["누적 관객"] = top_movies["누적 관객"].map(
                lambda x: f"{x:,.0f}"
            )
            top_movies["첫 주 관객"] = top_movies["첫 주 관객"].map(
                lambda x: f"{x:,.0f}"
            )
            top_movies["10위권 일수"] = top_movies["10위권 일수"].map(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
            )
            top_movies["롱런 지수"] = top_movies["롱런 지수"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "-"
            )

            top_movies.index = range(1, len(top_movies) + 1)

            st.dataframe(
                top_movies,
                use_container_width=True
            )

st.divider()

st.caption(
    "분석 방법: 속성 선택 → 표준화 → K-평균 군집화(K=3) → "
    "누적 관객 평균에 따라 유형 이름 지정. "
    "난수 시드 42를 사용해 같은 데이터와 설정에서 결과를 재현할 수 있습니다."
)
