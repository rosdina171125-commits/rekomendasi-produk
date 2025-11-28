import streamlit as st
import pandas as pd

# ======================
# CEK DAN IMPORT SKLEARN
# ======================
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    st.error("""
    ❌ **Library scikit-learn belum terinstall.**
    Jalankan perintah berikut pada terminal VS Code:

    ```
    pip install scikit-learn
    ```

    Setelah instalasi selesai, jalankan ulang aplikasi dengan:
    ```
    streamlit run app.py
    ```
    """)
    st.stop()

# ======================
# DATA PRODUK (contoh)
# ======================
data = [
    {
        "product_id": 1,
        "product_name": "Kopi Arabica Premium 250g",
        "category": "Minuman",
        "price": 65000,
        "rating": 4.8,
        "reviews": [
            "Rasanya enak dan aromanya kuat.",
            "Kopi berkualitas dan tidak terlalu asam.",
            "Tekstur halus dan tidak bikin sakit perut."
        ]
    },
    {
        "product_id": 2,
        "product_name": "Kopi Robusta Sachet",
        "category": "Minuman",
        "price": 15000,
        "rating": 4.1,
        "reviews": [
            "Harganya murah.",
            "Rasa lumayan untuk harian.",
            "Agak pahit tapi masih oke."
        ]
    },
    {
        "product_id": 3,
        "product_name": "Teh Hijau Organik",
        "category": "Minuman",
        "price": 30000,
        "rating": 4.5,
        "reviews": [
            "Rasanya segar.",
            "Cocok untuk diet.",
            "Aromanya lembut dan rileks."
        ]
    },
    {
        "product_id": 4,
        "product_name": "Snack Kentang Pedas",
        "category": "Makanan Ringan",
        "price": 12000,
        "rating": 4.2,
        "reviews": [
            "Pedas dan gurih.",
            "Teksturnya renyah.",
            "Enak buat teman nonton."
        ]
    },
    {
        "product_id": 5,
        "product_name": "Snack Keju Panggang",
        "category": "Makanan Ringan",
        "price": 17000,
        "rating": 4.7,
        "reviews": [
            "Keju terasa banget.",
            "Tidak terlalu asin.",
            "Cocok untuk camilan."
        ]
    },
]

# ====================================
# MEMBUAT DATAFRAME PRODUK + REVIEW
# ====================================
rows = []
for item in data:
    rows.append({
        "product_id": item["product_id"],
        "product_name": item["product_name"],
        "category": item["category"],
        "price": item["price"],
        "rating": item["rating"],
        "all_reviews": " ".join(item["reviews"])  # Gabung semua ulasan
    })

df_products = pd.DataFrame(rows)

# ====================================
# MODEL NLP: TF-IDF + COSINE SIMILARITY
# ====================================
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_products["all_reviews"])

def recommend(product_text, category=None, min_rating=0, top_k=5):
    # Ubah input user menjadi vektor
    user_vec = tfidf.transform([product_text])

    # Hitung similarity
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    df_temp = df_products.copy()
    df_temp["similarity"] = similarities

    # Filter kategori
    if category and category != "Semua":
        df_temp = df_temp[df_temp["category"] == category]

    # Filter rating
    df_temp = df_temp[df_temp["rating"] >= min_rating]

    # Sort
    df_temp = df_temp.sort_values("similarity", ascending=False)

    return df_temp.head(top_k)

# ====================================
# GUI STREAMLIT
# ====================================
st.set_page_config(page_title="Sistem Rekomendasi Produk", page_icon="🛒")

st.title("🧠 Sistem Rekomendasi Produk Berdasarkan Ulasan")
st.write("Aplikasi berbasis **Python + Streamlit + NLP (TF-IDF + Cosine Similarity)**.")

# Sidebar
st.sidebar.header("Pengaturan")
kategori_opsi = ["Semua"] + sorted(df_products["category"].unique().tolist())
selected_category = st.sidebar.selectbox("Kategori Produk", kategori_opsi)

min_rating = st.sidebar.slider(
    "Minimal Rating", 0.0, 5.0, 4.0, step=0.1
)

top_k = st.sidebar.slider(
    "Jumlah Rekomendasi", 1, 5, 3
)

# Input ulasan pengguna
st.subheader("📝 Masukkan Ulasan atau Preferensi Anda")
user_input = st.text_area(
    "Contoh: \"Saya ingin produk yang gurih dan tidak terlalu pedas\"",
    height=100
)

button = st.button("🔍 Cari Rekomendasi")

# Tampilkan hasil
if button:
    if user_input.strip() == "":
        st.warning("Silakan isi ulasan terlebih dahulu.")
    else:
        results = recommend(
            user_input,
            category=selected_category,
            min_rating=min_rating,
            top_k=top_k
        )

        if results.empty:
            st.error("Tidak ada produk yang cocok dengan filter.")
        else:
            st.success("Berhasil! Berikut rekomendasinya:")
            st.dataframe(
                results[["product_name", "category", "price", "rating", "similarity"]],
                use_container_width=True
            )

            st.bar_chart(results.set_index("product_name")["similarity"])

            st.subheader("Detail Produk:")
            for _, row in results.iterrows():
                st.markdown(f"""
                **{row['product_name']}**  
                Kategori: `{row['category']}`  
                Harga: Rp {row['price']:,}  
                Rating: ⭐ {row['rating']:.1f}  
                Similarity: `{row['similarity']:.3f}`  
                """)

# Footer
st.markdown("---")
st.caption("Dibuat untuk Proyek Akhir Pemrograman Visual II • Python + Streamlit")

