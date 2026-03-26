import requests
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

TMDB_API_KEY = "145a8c38c7b2182c288c6fde2be10905"

def get_genre_mapping():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url).json()
    return {genre['id']: genre['name'] for genre in response['genres']}

def fetch_latest_movies(pages=50): # 50 trang = 1000 phim mới nhất (Thích lấy 10.000 phim thì đổi thành 500)
    genre_map = get_genre_mapping()
    new_movies = []
    
    print(f"📥 Đang cào {pages} trang phim từ server TMDB...")
    for page in range(1, pages + 1):
        url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page={page}"
        response = requests.get(url).json()
        
        for movie in response.get('results', []):
            if not movie.get('title') or not movie.get('release_date') or not movie.get('genre_ids'):
                continue
            
            year = movie['release_date'][:4]
            title_with_year = f"{movie['title']} ({year})"
            genres = [genre_map.get(g_id) for g_id in movie['genre_ids'] if genre_map.get(g_id)]
            
            new_movies.append({
                'title': title_with_year,
                'genres': "|".join(genres)
            })
        # Ngủ 0.1s để không bị TMDB khóa IP vì spam request
        time.sleep(0.1) 
        
    return pd.DataFrame(new_movies)

# ==========================================
# LUỒNG CHẠY TỰ ĐỘNG (ETL PIPELINE)
# ==========================================
if __name__ == "__main__":
    # 1. Cào data mới
    df_new = fetch_latest_movies(pages=50) # Tải 1000 phim mới nhất
    
    # 2. Đọc bộ data 9.000 phim cũ của bạn
    print("🔄 Đang gộp với bộ dữ liệu cũ...")
    df_old = pd.read_csv('movies_with_clusters.csv')
    
    # 3. Gộp lại và XÓA TRÙNG (Chỉ giữ lại những phim thực sự mới)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset=['title'], keep='first', inplace=True)
    print(f"📈 Kích thước kho dữ liệu hiện tại: {len(df_combined)} bộ phim.")

    # 4. CHẠY LẠI NÃO AI (K-MEANS) CHO TOÀN BỘ PHIM
    print("🧠 Đang huấn luyện lại thuật toán K-Means cho dữ liệu mới...")
    # Mã hóa thể loại
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genres_matrix = vectorizer.fit_transform(df_combined['genres'])
    
    # Gom thành 10 cụm (như web cũ bạn đang làm)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    df_combined['Cluster'] = kmeans.fit_predict(genres_matrix)

    # 5. LƯU ĐÈ VÀO FILE WEB ĐANG ĐỌC
    df_combined.to_csv('movies_with_clusters.csv', index=False)
    print("✅ XONG! Hệ thống đã cập nhật AI và Dữ liệu mới nhất. Trang web đã sẵn sàng!")