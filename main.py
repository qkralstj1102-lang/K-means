
import math

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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

CLUSTER_LABELS = ["㉮", "㉯", "㉰"]


# ==================================================
# 2. 제목
# ==================================================
st.title("🎬 영화 유형 나누기")
st.caption(
    "영화 데이터를 분석해 비슷한 특성을 가진 영화들을 "
    "K-평균 알고리즘으로 세 가지 유형으로 나눕니다."
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

    # 숫자형 데이터로 변환할 열
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

    # 영화 제목이 없는 행 제외
    df = df.dropna(subset=["movieNm"]).copy()

    # 데이터에 들어 있는 전체 영화 편수
    total_count = len(df)

    # 스크린 수와 누적 관객에 상용로그 적용
    # 로그는 양수에 대해서만 계산
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
    # 첫 주 관객이 0이면 분석에서 제외
    df["longrun_index"] = float("nan")

    valid_first_week = (
        df["first_week_audi"].notna()
        & (df["first_week_audi"] > 0)
    )

    df.loc[valid_first_week, "longrun_index"] = (
        df.loc[valid_first_week, "total_audi"]
        / df.loc[valid_first_week, "first_week_audi"]
    )

    # 롱런 지수는 최대 20
    df["longrun_index"] = df["longrun_index"].clip(upper=20)

    # 네 가지 속성 중 결측치가 있으면 제외
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

    clean_df = df.loc[valid_mask].copy()

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
    "네 가지 속성 중 결측치가 있는 영화, 첫 주 관객이 0인 영화, "
    "로그 계산이 불가능한 영화는 분석에서 제외합니다."
)

if len(df) < 3:
    st.error("군집화를 하려면 분석 가능한 영화가 최소 3편 필요합니다.")
    st.stop()

st.divider()


# ==================================================
# 4. 군집화 속성 선택
# ==================================================
st.subheader("1. 영화 유형을 나눌 기준")

selected_features = st.multiselect(
    "군집화에 사용할 속성을 선택하세요.",
    options=list(FEATURES.keys()),
    default=list(FEATURES.keys()),
    help="두 개 이상 선택해야 합니다. 기본값은 네 가지 속성 전체입니다.",
)

if len(selected_features) < 2:
    st.warning("군집화 속성을 두 개 이상 선택해 주세요.")
    st.stop()

feature_columns = [
    FEATURES[name] for name in selected_features
]


# ==================================================
# 5. 표준화 및 K-평균 군집화
# ==================================================
X = df[feature_columns].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 높은 군집부터 이름 부여
cluster_means = (
    df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
)

cluster_order = cluster_means.index.tolist()

label_map = {
    cluster_id: CLUSTER_LABELS[index]
    for index, cluster_id in enumerate(cluster_order)
}

df["유형"] = df["cluster_raw"].map(label_map)

st.info(
    "군집 이름은 누적 관객 평균이 높은 순서로 "
    "㉮, ㉯, ㉰를 부여했습니다."
)

st.divider()


# ==================================================
# 6. 2차원 산점도
# ==================================================
st.subheader("2. 2차원 산점도")

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
    category_orders={"유형": CLUSTER_LABELS},
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
    "점 하나가 영화 한 편입니다. 점 위에 마우스를 올리면 "
    "영화 제목과 속성 값을 확인할 수 있습니다."
)

st.divider()


# ==================================================
# 7. 3차원 산점도
# ==================================================
st.subheader("3. 3차원 산점도")

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
        category_orders={"유형": CLUSTER_LABELS},
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
        "점 위에 마우스를 올리면 영화 제목을 확인할 수 있습니다."
    )

st.divider()


# ==================================================
# 8. 유형별 통계
# ==================================================
st.subheader("4. 유형별 통계")

# 한글 별칭 대신 영문 이름을 사용해 문법 오류 방지
stats = (
    df.groupby("유형")
    .agg(
        movie_count=("movieNm", "count"),
        avg_first_scrn=("first_scrn", "mean"),
        avg_total_audi=("total_audi", "mean"),
        avg_days_in_top10=("days_in_top10", "mean"),
        avg_longrun_index=("longrun_index", "mean"),
    )
    .reindex(CLUSTER_LABELS)
)

# 화면에 표시할 한글 열 이름
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
    "스크린 수와 누적 관객은 로그 변환 전의 원래 단위로 평균을 계산합니다. "
    "롱런 지수는 영화별 계산값을 사용합니다."
)

st.divider()


# ==================================================
# 9. 유형별 누적 관객 TOP 5
# ==================================================
st.subheader("5. 유형별 대표 영화 TOP 5")

for label in CLUSTER_LABELS:
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
# 10. 분석 안내
# ==================================================
st.subheader("분석 방법")

st.markdown(
    """
    1. **데이터 전처리:** 분석 속성에 결측치가 있는 영화 등을 제외합니다.
    2. **속성 변환:** 스크린 수와 누적 관객에 상용로그를 적용합니다.
    3. **롱런 지수 계산:** 누적 관객을 첫 주 관객으로 나누고 최대 20으로 제한합니다.
    4. **표준화:** 선택한 속성의 평균과 표준편차를 이용해 표준화합니다.
    5. **군집화:** K-평균 알고리즘으로 영화들을 세 유형으로 분류합니다.
    6. **유형 이름 지정:** 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰로 표시합니다.
    """
)

st.caption(
    "같은 데이터와 같은 속성을 선택하면 random_state=42 설정으로 "
    "군집화 결과를 재현할 수 있습니다."
)
