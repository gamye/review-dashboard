import streamlit as st
import pandas as pd
import re
from konlpy.tag import Okt
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from wordcloud import WordCloud
import altair as alt

# --------------------------------------------------------------------------
# 1. 초기 설정 및 데이터 불러오기
# --------------------------------------------------------------------------

st.set_page_config(layout="wide")

@st.cache_data
def load_data(file_path):
    """CSV 파일을 불러오고 기본 처리를 수행합니다."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일이 현재 폴더에 있는지 확인해주세요.")
        return None
    
    if '리뷰작성일시' in df.columns:
        df.rename(columns={'리뷰작성일시': '작성일'}, inplace=True)
    else:
        st.error("오류: 날짜 정보를 담고 있는 '리뷰작성일시' 열을 찾을 수 없습니다.")
        return None

    df['작성일'] = pd.to_datetime(df['작성일'])
    df['년'] = df['작성일'].dt.year
    df['월'] = df['작성일'].dt.month
    df['연월'] = df['작성일'].dt.strftime('%Y-%m')
    
    if '리뷰평점' in df.columns:
        df.rename(columns={'리뷰평점': '평점'}, inplace=True)
    else:
        st.error("오류: '리뷰평점' 열을 찾을 수 없습니다.")
        return None

    df['리뷰내용'] = df['리뷰내용'].fillna('')
    df['리뷰내용_전처리'] = df['리뷰내용'].apply(preprocess_text)
    
    return df

@st.cache_data
def preprocess_text(text):
    """텍스트에서 불필요한 부분을 제거하는 함수"""
    text = re.sub(r'\(\d{4}-\d{2}-\d{2}.*에 등록된 네이버 페이 구매평\)', '', text)
    text = re.sub(r'[^가-힣\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_korean_font_path():
    import os
    if platform.system() == 'Windows':
        return 'c:/Windows/Fonts/malgun.ttf'
    elif platform.system() == 'Darwin':
        return '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
    else: # 리눅스 (Streamlit Cloud)
        nanum_font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        if os.path.exists(nanum_font_path):
            return nanum_font_path
        else:
            st.warning("나눔고딕 글꼴을 찾을 수 없어 기본 글꼴로 표시됩니다. (한글 깨짐 가능성)")
            return None

FONT_PATH = get_korean_font_path()

if FONT_PATH:
    import matplotlib.font_manager as fm
    if not any(f.name == 'NanumGothic' for f in fm.fontManager.ttflist):
        fm.fontManager.addfont(FONT_PATH)
    plt.rc('font', family='NanumGothic')
    
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
    
plt.rcParams['axes.unicode_minus'] = False

# --------------------------------------------------------------------------
# 2. 분석 및 화면 표시용 함수
# --------------------------------------------------------------------------
@st.cache_resource
def get_okt():
    return Okt()

okt = get_okt()
stopwords = [
    '구매', '착용', '생각', '선물', '주문', '배송', '느낌', '제품', '상품', '사은', '팔찌',
    '마음', '부분', '조금', '이거', '저', '것', '수', '좀', '그', '더', '그냥', '하니',
    '해서', '같아요', '입니다', '하고', '했는데', '있고', '정말', '너무', '많이', '보고', '아주', '하나', '일만', '듭니'
]

@st.cache_data
def get_keywords(text_series):
    """주어진 텍스트에서 명사를 추출하여 목록으로 반환"""
    all_nouns = []
    for review in text_series:
        nouns = okt.nouns(review)
        filtered_nouns = [word for word in nouns if len(word) > 1 and word not in stopwords]
        all_nouns.extend(filtered_nouns)
    return all_nouns

def display_wordcloud(keyword_counts, title):
    if not keyword_counts:
        st.info(f"'{title}'에 대한 자료가 없어 구름 그림을 표시할 수 없습니다.")
        return
    wc = WordCloud(font_path=FONT_PATH, background_color='white', width=400, height=250, colormap='viridis').generate_from_frequencies(dict(keyword_counts))
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    st.pyplot(fig)

# --------------------------------------------------------------------------
# 3. 대시보드 화면 구성
# --------------------------------------------------------------------------

st.title('📈 리뷰 데이터 분석 대시보드')

df_origin = load_data('240611-250611_Quick_Review_Filter.csv')

if df_origin is not None:
    # --- [수정됨] 옆쪽 메뉴 (사이드바) 구성 변경 ---
    st.sidebar.header('🗓️ 기간 필터')
    
    year_list = sorted(df_origin['년'].unique(), reverse=True)
    selected_year = st.sidebar.selectbox('연도 선택', year_list)

    month_list = ['전체'] + sorted(df_origin[df_origin['년'] == selected_year]['월'].unique())
    selected_month = st.sidebar.selectbox('월 선택', month_list)

    st.sidebar.divider()
    st.sidebar.header('⚙️ 분석 기준 설정')
    
    # 1. m값 자동 계산 (전체 데이터의 리뷰 개수 중앙값)
    review_counts_per_product = df_origin.groupby('상품명')['상품명'].count()
    # 데이터가 하나도 없을 경우를 대비해 예외 처리
    if not review_counts_per_product.empty:
        m_default = int(review_counts_per_product.median())
        max_reviews = int(review_counts_per_product.max())
    else:
        m_default = 10 # 기본값
        max_reviews = 100 # 기본값
    
    # 2. 최소 리뷰 개수 슬라이더
    min_review_count = st.sidebar.slider(
        '최소 리뷰 개수 필터', 
        min_value=0, 
        max_value=max_reviews, 
        value=0, # 기본값은 0으로 설정하여 모든 상품을 보여줌
        step=1
    )
    st.sidebar.info(f"💡 '신뢰도 점수' 계산 시 사용된 가중치(m)의 기본값은 리뷰 개수의 중앙값인 **{m_default}개**입니다.")


    # --- 데이터 선택하기 ---
    df_year_filtered = df_origin[df_origin['년'] == selected_year]
    
    if selected_month == '전체':
        df_filtered = df_year_filtered
    else:
        df_filtered = df_year_filtered[df_year_filtered['월'] == selected_month]

    # --- 1. 연간 평균 평점 추이 ---
    st.header(f'📅 {selected_year}년 전체 평점 추이')
    
    yearly_rating_trend = df_year_filtered.groupby('월')['평점'].mean().reset_index()
    yearly_rating_trend['월'] = yearly_rating_trend['월'].astype(str) + '월'
    
    trend_chart = alt.Chart(yearly_rating_trend).mark_line(point=True).encode(
        x=alt.X('월', sort=None, title='월'),
        y=alt.Y('평점', scale=alt.Scale(zero=False), title='평균 평점'),
        tooltip=['월', '평점']
    ).properties(
        title=f'{selected_year}년 월별 평균 평점'
    )
    st.altair_chart(trend_chart, use_container_width=True)
    st.divider()

    if not df_filtered.empty:
        st.header(f'📊 {selected_year}년 {selected_month if selected_month != "전체" else "전체"}월 분석')

        # --- [수정됨] 2. 상품별 상세 분석 (신뢰도 점수 적용) ---
        st.subheader('📦 상품별 상세 분석')

        # 선택된 기간의 전체 평균 평점 (C값)
        C = df_filtered['평점'].mean()
        m = m_default # m값은 중앙값으로 설정

        rating_df = df_filtered.groupby('상품명').agg(
            평균평점=('평점', 'mean'),
            리뷰수=('평점', 'count')
        ).reset_index()

        # 신뢰도 점수 계산
        v = rating_df['리뷰수']
        R = rating_df['평균평점']
        rating_df['신뢰도점수'] = (v / (v + m)) * R + (m / (v + m)) * C
        
        # 최소 리뷰 개수 필터 적용
        rating_df_filtered = rating_df[rating_df['리뷰수'] >= min_review_count]

        # 신뢰도 점수가 낮은 순으로 정렬
        rating_df_sorted = rating_df_filtered.sort_values(by='신뢰도점수', ascending=True)

        # 화면에 표시할 열 순서 재배치
        display_cols = ['상품명', '리뷰수', '평균평점', '신뢰도점수']
        st.dataframe(
            rating_df_sorted[display_cols].style
            .background_gradient(cmap='Reds_r', subset=['평균평점', '신뢰도점수'])
            .format({'평균평점': '{:.2f}', '신뢰도점수': '{:.2f}'}),
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("####  특정 상품 리뷰 원문 보기 (핵심어 강조)")
        # 필터링된 상품 목록을 드롭다운에 사용
        product_list = ["상품을 선택하세요"] + rating_df_sorted['상품명'].tolist()
        selected_product = st.selectbox('상품을 선택하면 해당 상품의 리뷰 원문을 보여줍니다.', product_list)

        if selected_product != "상품을 선택하세요":
            product_reviews = df_filtered[df_filtered['상품명'] == selected_product].copy()
            
            avg_rating = product_reviews['평점'].mean()
            review_count = len(product_reviews)
            st.metric(label=f"'{selected_product}' 평균 평점", value=f"{avg_rating:.2f} 점", delta=f"리뷰 {review_count}개")

            product_keywords_list = get_keywords(product_reviews['리뷰내용_전처리'])

            if product_keywords_list:
                product_keyword_counts = Counter(product_keywords_list)
                top_keywords = [kw for kw, count in product_keyword_counts.most_common(5)]
                st.info(f"💡 이 상품의 주요 핵심어: **{', '.join(top_keywords)}** (리뷰 내용에서 빨간색으로 강조됩니다)")

                def highlight_keywords(text, keywords):
                    for keyword in keywords:
                        text = re.sub(f'({re.escape(keyword)})', r'<span style="color: red; font-weight: bold;">\1</span>', text)
                    return text

                display_df = product_reviews[['평점', '리뷰내용']].sort_values(by='평점', ascending=True)
                display_df['리뷰내용'] = display_df['리뷰내용'].apply(lambda x: highlight_keywords(x, top_keywords))
                
                st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

            else:
                st.info(f'"{selected_product}" 상품에 대한 분석 가능한 핵심어가 없습니다. 원본 리뷰만 표시합니다.')
                st.dataframe(product_reviews[['평점', '리뷰내용']].sort_values(by='평점', ascending=True), use_container_width=True)
        
        st.divider()

        # --- 5. 전체 핵심어 경향 분석 ---
        st.subheader('💬 전체 핵심어 경향 분석')
        
        tab1, tab2 = st.tabs(['월별 핵심어 경향', '긍정/부정 핵심어 분석'])

        with tab1:
            st.markdown(f"#### {selected_year}년 전체 상위 10개 핵심어 월별 빈도 추이")
            
            total_year_keywords = get_keywords(df_year_filtered['리뷰내용_전처리'])
            top10_keywords = [kw for kw, count in Counter(total_year_keywords).most_common(10)]

            monthly_keywords_list = []
            for month in sorted(df_year_filtered['월'].unique()):
                month_df = df_year_filtered[df_year_filtered['월'] == month]
                if not month_df.empty:
                    month_keywords = get_keywords(month_df['리뷰내용_전처리'])
                    month_counts = Counter(month_keywords)
                    for keyword in top10_keywords:
                        monthly_keywords_list.append({'월': f'{month}월', '핵심어': keyword, '빈도수': month_counts[keyword]})
            
            if monthly_keywords_list:
                trend_df = pd.DataFrame(monthly_keywords_list)
                
                keyword_trend_pivot = trend_df.pivot_table(index='핵심어', columns='월', values='빈도수', fill_value=0)
                keyword_trend_pivot = keyword_trend_pivot.reindex(top10_keywords)
                
                month_order = [f'{i}월' for i in range(1, 13)]
                ordered_columns = [month for month in month_order if month in keyword_trend_pivot.columns]
                keyword_trend_pivot = keyword_trend_pivot[ordered_columns]
                
                st.dataframe(keyword_trend_pivot.style.background_gradient(cmap='viridis').format('{:.0f}'), use_container_width=True)
            else:
                st.info("핵심어 경향 자료를 생성할 수 없습니다.")

        with tab2:
            st.markdown(f"#### {selected_month if selected_month != '전체' else '전체'}월 긍정/부정 핵심어")
            
            col1, col2 = st.columns(2)
            
            positive_reviews_text = df_filtered[df_filtered['평점'] >= 4]['리뷰내용_전처리']
            positive_keywords_list = get_keywords(positive_reviews_text)
            
            negative_reviews_text = df_filtered[df_filtered['평점'] <= 3]['리뷰내용_전처리']
            negative_keywords_list = get_keywords(negative_reviews_text)

            with col1:
                display_wordcloud(Counter(positive_keywords_list), '긍정 리뷰 (4-5점)')
            with col2:
                display_wordcloud(Counter(negative_keywords_list), '부정 리뷰 (1-3점)')

    else:
        st.warning(f"{selected_year}년 {selected_month}월에는 리뷰 자료가 없습니다.")