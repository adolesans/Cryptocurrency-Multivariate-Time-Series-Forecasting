# 📈 Bitcoin Price Forecasting: Multi-Step Time Series with Seq2Seq LSTM
Proyek ini merupakan submisi akhir untuk kelas **Membangun Proyek Deep Learning Tingkat Mahir**. Setelah mempelajari fundamental TensorFlow, kustomisasi arsitektur, hingga pelatihan model tanpa fungsi *built-in*, proyek ini bertujuan untuk mendemonstrasikan kemampuan tersebut dalam kasus nyata yang kompleks.

Tantangan utama dalam proyek ini adalah melakukan **Multi-Step Forecasting** (memprediksi 24 jam ke depan sekaligus) pada data harga Bitcoin yang memiliki volatilitas tinggi. Metode standar seringkali kurang memadai untuk menangkap dinamika jangka panjang, sehingga diperlukan pendekatan arsitektur yang lebih canggih.

## 🎯 Objectives
1.  Membangun **Custom Model Architecture** menggunakan pendekatan Sequence-to-Sequence (Seq2Seq).
2.  Mengimplementasikan **Custom Training Loop** menggunakan `tf.GradientTape` untuk kontrol penuh atas proses pembelajaran (menggantikan `model.fit`).
3.  Menerapkan teknik **Autoregressive Inference** untuk prediksi multi-horizon.
4.  Membandingkan performa model *Advanced* (Seq2Seq) dengan model *Baseline* (Vanilla LSTM).

## 📂 Dataset
* **Sumber Data:** Hourly Bitcoin Price (BTC-USD).
* **Fitur:** `Close` (Target), `Volume`, `RSI`, `MACD`, `Rolling Mean`, `Rolling Std`.
* **Windowing:** Input masa lalu 24 jam (`INPUT_WIDTH=24`) untuk memprediksi 24 jam ke depan (`OUT_STEPS=24`).

## 🛠️ Technical Implementation 
Proyek ini menggunakan teknik *Deep Learning* tingkat lanjut sesuai kriteria kelulusan:

### 1. Model Architecture: Seq2Seq LSTM
Berbeda dengan model sekuensial biasa, arsitektur ini menggunakan dua bagian utama:
* **Encoder:** Memproses urutan input historis dan merangkumnya menjadi *Context Vector*.
* **Decoder:** Menggunakan *Context Vector* tersebut untuk menghasilkan prediksi langkah demi langkah (*step-by-step*).

### 2. Custom Training Loop
Alih-alih menggunakan fungsi standar Keras, pelatihan dilakukan secara manual:
* Menggunakan **`tf.GradientTape`** untuk menghitung gradien.
* Menerapkan **Teacher Forcing** pada Decoder untuk mempercepat konvergensi saat pelatihan.
* Optimasi loss function (MAE) secara eksplisit per *batch*.

### 3. Inference Mechanism
* **Baseline:** *One-shot prediction* (memprediksi vektor 24 jam sekaligus).
* **Seq2Seq:** *Autoregressive* (output langkah $t$ menjadi input untuk langkah $t+1$).

## 📊 Results & Analysis
Berdasarkan evaluasi pada data uji (Test Set):
* **Baseline Model LSTM:** Cenderung menghasilkan prediksi yang kasar (*noisy*) dan kurang akurat dalam jangka panjang.
* **Seq2Seq Model:** Menghasilkan kurva prediksi yang lebih halus (*smooth*) dan mampu mempertahankan struktur tren global dengan lebih baik, meskipun terdapat tantangan *compounding error* pada langkah waktu akhir.

**Kesimpulan:** Model Seq2Seq dengan Custom Training terbukti lebih *robust* untuk menangani kompleksitas data *time series* Bitcoin dibandingkan pendekatan standar.

## 🚀 How to Run
1.  **Environment Setup:**
    Pastikan library terinstall dengan menjalankan:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Notebook:**
    Buka file `Annisa_Dewiyanti_Submission_Akhir_DLTM.ipynb` di Jupyter Notebook atau Google Colab dan jalankan seluruh sel (Run All).

## 📂 Project Structure
```text
DLTM_Annisa-Dewiyanti/
├── Annisa_Dewiyanti_Submission_Akhir_DLTM.ipynb  # Notebook Utama (Code)
├── model_baseline_LSTM.keras                     # Model Pembanding
├── model_seq2seq_LSTM.keras                      # Model Utama (Advanced)
├── requirements.txt                              # Dependencies
└── README.md                                     # Dokumentasi Proyek
