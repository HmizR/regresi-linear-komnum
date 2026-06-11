# Tugas Pemrograman Komputasi Numerik
</div>

|    NRP     |           Nama             |
| :--------: |       :------------:       |
| 5025251246 | Hamizan Rifqi Afandi       |
| 5025251247 | Gede Panji Dana Putra Ricedes |
| 5025251248 | Maulana Bagas Rizqi Pratama  |
| 5025251249 | Enver Alif Wirawan |

# Visualisasi Regresi Linear Sederhana
</div>

Aplikasi web berbasis Streamlit untuk melakukan analisis regresi linear sederhana secara interaktif. Pengguna dapat memasukkan data secara manual atau melalui file CSV, melakukan analisis regresi, mengevaluasi asumsi model, melakukan prediksi, serta mengunduh hasil analisis.

## Fitur
</div>

- Input data secara manual.
- Upload dataset CSV.
- Dataset contoh bawaan.
- Editor data interaktif.
- Perhitungan regresi linear sederhana.
- Visualisasi scatter plot dan garis regresi.
- Confidence Interval (CI) 95%.
- Prediction Interval (PI) 95%.
- Tabel koefisien regresi.
- Interval kepercayaan parameter.
- Diagnostik residual:
  - Residual vs X
  - Q-Q Plot
  - Fitted vs Residual
  - Uji normalitas Shapiro-Wilk
  - Skewness dan kurtosis
- Prediksi nilai Y untuk nilai X baru.
- Analisis ANOVA.
- Statistik deskriptif.
- Download hasil dalam format CSV, TXT, dan PNG.

## Tampilan Aplikasi
</div>

Berikut adalah tampilan aplikasi web dengan menggunakan contoh input
| Variabel | Data |
| :--- | :--- |
| x | 1, 2, 3, 4, 5 |
| y | 2, 3, 4, 5, 6 |
### Halaman Regresi

![image alt](https://github.com/HmizR/regresi-linear-komnum/blob/0e80543cdb488b3282196d168567918d8ae67f20/Regresi.png)

### Diagnostik Residual

![image alt](https://github.com/HmizR/regresi-linear-komnum/blob/0e80543cdb488b3282196d168567918d8ae67f20/Diagnostik.png)

### Prediksi

![image alt](https://github.com/HmizR/regresi-linear-komnum/blob/0e80543cdb488b3282196d168567918d8ae67f20/Prediksi.png)

## Panduan Input Data
</div>

Aplikasi web ini menyediakan tiga metode input:

### 1. Dataset Contoh
Pilih salah satu dataset bawaan:
- Kecil (n=5)
- Sedang (n=10)
- Dengan Noise

### 2. Input Manual
Masukkan data x dan y menggunakan contoh format:

x:
1, 2, 3, 4, 5

y:
2, 4, 5, 4, 6

Dengan menggunakan pemisah berupa koma atau spasi.

### 3. Upload CSV
Unggah file CSV dengan contoh format:

x,y
1,2
2,4
3,5
4,4
5,6

## Output Program
</div>
Berdasarkan input aplikasi web akan menghasilkan output dengan mengaplikasikan rumus regresi linear, berikut adalah beberapa jenis output yang dihasilkan:

### Regresi
- Persamaan regresi.
- R-squared.
- Adjusted R-squared.
- Standard Error.

### Diagnostik
- Residual Plot.
- Q-Q Plot.
- Fitted vs Residual.
- Uji Shapiro-Wilk.

### Prediksi
- Nilai prediksi Y.
- Confidence Interval 95%.
- Prediction Interval 95%.

### Statistik
- ANOVA.
- Korelasi Pearson.
- Statistik deskriptif.

### Download
- CSV hasil prediksi.
- TXT ringkasan regresi.
- PNG visualisasi regresi.
