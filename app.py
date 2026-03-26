import streamlit as st
import pandas as pd
import requests
import re
from streamlit_option_menu import option_menu
import plotly.express as px
from sklearn.decomposition import PCA

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & CSS
# ==========================================
TMDB_API_KEY = "145a8c38c7b2182c288c6fde2be10905" 

# Ép thanh menu bên trái luôn mở khi load trang
st.set_page_config(page_title="Hệ thống Gợi ý Phim", layout="wide", page_icon="🎬", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Xóa bỏ các menu thừa của Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* LƯU Ý: Đã xóa dòng ẩn header để không bị mất nút mũi tên mở Menu */
    
    /* Đổi màu nền toàn trang thành Gradient tối */
    .stApp {
        background: radial-gradient(circle at 20% 50%, #141414 0%, #000000 100%);
        color: #E5E5E5;
    }
    
    /* Hiệu ứng Hover cho ảnh Poster */
    img {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-radius: 8px; 
    }
    img:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(229, 9, 20, 0.4); 
    }
    
    /* Chỉnh lại font chữ và màu các tiêu đề */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_title):
    clean_title = re.sub(r'\s*\(\d{4}\)\s*$', '', movie_title)
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={clean_title}&language=vi-VN"
    
    details = {
        "poster": "https://via.placeholder.com/500x750/333333/FFFFFF?text=No+Poster",
        "backdrop": "https://via.placeholder.com/1280x720/333333/FFFFFF?text=No+Backdrop",
        "rating": "N/A",
        "year": "N/A",
        "overview": "Chưa có thông tin tóm tắt cho bộ phim này trên TMDB."
    }
    
    try:
        response = requests.get(url)
        data = response.json()
        if data['results']:
            movie = data['results'][0]
            if movie.get('poster_path'):
                details["poster"] = "https://image.tmdb.org/t/p/w500" + movie['poster_path']
            if movie.get('backdrop_path'):
                details["backdrop"] = "https://image.tmdb.org/t/p/w1280" + movie['backdrop_path']
            
            details["rating"] = round(movie.get('vote_average', 0), 1)
            details["year"] = movie.get('release_date', 'N/A')[:4]
            if movie.get('overview'):
                details["overview"] = movie.get('overview')
    except:
        pass
        
    return details

@st.cache_data
def load_data():
    df = pd.read_csv('movies_with_clusters.csv')
    df['year_extracted'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
    return df

movies = load_data()

# ==========================================
# 3. THANH ĐIỀU HƯỚNG (SIDEBAR MENU)
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150) 
    st.markdown("<br>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None, 
        options=["Khám phá Phim", "Không gian Thuật toán", "Thông tin Đồ án"], 
        icons=["film", "cpu", "person-vcard"], 
        menu_icon="cast", 
        default_index=0, 
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#E50914", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#333"},
            "nav-link-selected": {"background-color": "#E50914", "color": "white"}, 
        }
    )

# ==========================================
# 4. TRANG 1: KHÁM PHÁ PHIM
# ==========================================
if selected == "Khám phá Phim":
    st.title("🎬 Hệ Thống Gợi Ý Phim")
    st.subheader("Khám phá vũ trụ điện ảnh bằng AI")
    
    selected_movie = st.selectbox("Gõ tên một bộ phim (tiếng Anh):", movies['title'].values)

    if st.button("Gợi ý cho tôi"):
        movie_cluster = movies[movies['title'] == selected_movie].iloc[0]['Cluster']
        details = fetch_movie_details(selected_movie)
        
        st.markdown("---")
        
        # --- HERO BANNER ---
        st.image(details["backdrop"], use_container_width=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(details["poster"], use_container_width=True)
        with col2:
            st.markdown(f"## {selected_movie}")
            st.markdown(f"**⭐ Điểm IMDb:** {details['rating']}/10 &nbsp;&nbsp;|&nbsp;&nbsp; **📅 Năm:** {details['year']}")
            st.write(f"**📖 Nội dung:** {details['overview']}")
            st.info(f"🧠 K-Means đã xếp bộ phim này vào nhóm đặc trưng: **Cluster {movie_cluster}**")
            
            # --- NÚT XEM TRAILER ---
            youtube_search_url = f"https://www.youtube.com/results?search_query={selected_movie.replace(' ', '+')}+official+trailer"
            if hasattr(st, 'link_button'):
                st.link_button("▶ Xem Trailer", youtube_search_url, type="primary")
            else:
                st.markdown(f'<a href="{youtube_search_url}" target="_blank" style="background-color:#E50914;color:white;padding:10px 24px;border-radius:4px;text-decoration:none;font-weight:bold;display:inline-block;margin-top:10px;">▶ Xem Trailer</a>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- MULTI-ROW GỢI Ý ---
        similar_movies = movies[movies['Cluster'] == movie_cluster]
        similar_movies = similar_movies[similar_movies['title'] != selected_movie] 
        
        def draw_movie_row(row_title, movie_list):
            if movie_list.empty: return
            st.markdown(f"### {row_title}")
            sample_movies = movie_list.sample(n=min(5, len(movie_list)))
            cols = st.columns(5)
            for i, (_, row_data) in enumerate(sample_movies.iterrows()):
                rec_details = fetch_movie_details(row_data['title']) 
                with cols[i]:
                    st.image(rec_details["poster"], use_container_width=True)
                    st.write(f"**{row_data['title']}**")
                    st.caption(f"Năm: {int(row_data['year_extracted']) if pd.notna(row_data['year_extracted']) else 'N/A'}")

        with st.spinner("Đang phân tích dữ liệu và tải Poster độ phân giải cao..."):
            draw_movie_row("🔥 Khám phá các siêu phẩm cùng thể loại", similar_movies)
            
            classic_movies = similar_movies[similar_movies['year_extracted'] < 2000]
            draw_movie_row("📼 Tuyệt tác hoài cổ (Trước 2000)", classic_movies)
            
            modern_movies = similar_movies[similar_movies['year_extracted'] >= 2000]
            draw_movie_row("🚀 Siêu phẩm thế kỷ 21", modern_movies)

# ==========================================
# 5. TRANG 2: KHÔNG GIAN THUẬT TOÁN
# ==========================================
elif selected == "Không gian Thuật toán":
    st.title("🌌 Phân tích Không gian Phân cụm")
    st.markdown("""
    Biểu đồ thể hiện cách thuật toán K-Means gom hơn 9.000 bộ phim thành các cụm (Clusters). 
    Sử dụng kỹ thuật **PCA (Principal Component Analysis)** để nén 20 chiều thể loại xuống không gian đồ họa 3D.
    *👉 Bạn có thể dùng chuột để xoay, phóng to/thu nhỏ và chọn từng cụm.*
    """)
    st.markdown("---")
    
    with st.spinner("Đang render không gian 3D..."):
        genres_matrix = movies['genres'].str.get_dummies(sep='|')
        
        pca = PCA(n_components=3)
        components = pca.fit_transform(genres_matrix)
        
        movies['x'] = components[:, 0]
        movies['y'] = components[:, 1]
        movies['z'] = components[:, 2]
        
        movies['Nhãn Cụm'] = "Cluster " + movies['Cluster'].astype(str)
        
        fig = px.scatter_3d(
            movies, x='x', y='y', z='z',
            color='Nhãn Cụm',
            hover_name='title', 
            hover_data={'Nhãn Cụm': True, 'genres': True, 'x': False, 'y': False, 'z': False},
            opacity=0.6,
            height=800
        )
        
        fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. TRANG 3: THÔNG TIN ĐỒ ÁN
# ==========================================
elif selected == "Thông tin Đồ án":
    st.title("👨‍💻 Về Đồ án này")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4140/4140048.png", width=200)
    
    with col2:
        st.subheader("Thông tin Sinh viên")
        st.write("👉 **Họ và tên:** Vinh")
        st.write("👉 **Trường:** Học viện Công nghệ Bưu chính Viễn thông (PTIT)")
        st.write("👉 **Niên khóa:** Sinh viên năm 3 - Chuyên ngành CNTT")
        
        st.subheader("Mục tiêu kỹ thuật")
        st.write("- Áp dụng thuật toán Học máy không giám sát (K-Means Clustering).")
        st.write("- Ứng dụng kỹ thuật One-Hot Encoding và PCA (Principal Component Analysis).")
        st.write("- Xây dựng Web App tương tác với Streamlit và tích hợp RESTful API (TMDB).")