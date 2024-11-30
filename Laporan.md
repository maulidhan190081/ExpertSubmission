# Laporan Proyek Machine Learning - Maulidhan Ady Nugraha

## Domain Proyek

Pasar properti di California dan Indonesia memiliki perbedaan yang signifikan dalam hal harga dan dinamika pasar. Di California, harga rumah sangat tinggi, dipengaruhi oleh pertumbuhan ekonomi yang pesat dan permintaan yang tinggi di kota-kota besar seperti Los Angeles dan San Francisco. Sebaliknya, di Indonesia, meskipun harga rumah lebih terjangkau, pasar perumahan di kota besar seperti Jakarta terus berkembang, didorong oleh urbanisasi dan pertumbuhan kelas menengah.

Proyek ini menggunakan data California Housing Prices untuk membangun model prediksi harga rumah, yang dapat memberikan wawasan tentang faktor-faktor yang memengaruhi harga properti di pasar yang berbeda. Hasil dari proyek ini diharapkan membantu pemahaman lebih dalam tentang dinamika harga rumah, baik di California maupun di Indonesia, sehingga memberikan referensi yang berguna untuk keputusan investasi di pasar properti Indonesia.

Perbedaan harga rumah antara California dan Indonesia juga dipengaruhi oleh faktor-faktor seperti kebijakan pemerintah dan infrastruktur. Di California, harga rumah didorong oleh permintaan yang sangat tinggi di daerah perkotaan dan semakin terbatasnya lahan, sehingga memicu kenaikan harga yang signifikan. Di Indonesia, meskipun harga rumah masih relatif lebih terjangkau, tantangan seperti ketersediaan lahan di kota-kota besar dan pengaruh kebijakan pemerintah yang mendorong pembangunan perumahan bagi masyarakat berpenghasilan rendah, tetap memengaruhi fluktuasi harga. Dengan menggunakan data dari California, model prediksi ini dapat memberikan gambaran tentang bagaimana faktor-faktor serupa mungkin memengaruhi pasar perumahan di Indonesia, meski dengan konteks yang berbeda.

## Business Understanding
Proses klarifikasi masalah bertujuan untuk memberikan landasan yang kuat dalam mengidentifikasi dan memahami permasalahan utama serta menyusun strategi untuk menyelesaikannya. Berikut ini adalah penjelasan rinci:

### Problem Statements
1.  Variasi Harga Rumah yang Sulit Diprediksi.
   
      Harga rumah sangat dipengaruhi oleh berbagai faktor seperti lokasi, fasilitas, ukuran properti, dan tren pasar. Kompleksitas hubungan antarvariabel ini membuat prediksi harga menjadi tantangan, terutama di pasar yang dinamis.

2. Mencari metode terbaik 
      
      Metode yang disarankan untuk mengatasi masalah prediksi harga rumah adalah LSTM dan GRU karena sudah mampu menjawab masalah prediksi data numerik [(Hochreiter & Schmidhuber, 1997; Greff et al., 2017)](https://doi.org/10.1162/neco.1997.9.8.1735);[Cho et al. (2014)](https://aclanthology.org/D14-1179).

### Goals
1. Mengembangkan model prediktif berbasis teknologi modern seperti LSTM dan GRU yang mampu menangkap pola kompleks dan hubungan non-linear dalam data harga rumah.

2. Memperoleh metode terbaik antara algoritma deep learning (LSTM dan GRU) yang sudah secara khusus dirancang untuk menangani data yang beragam dan sangat disarankan untuk prediksi harga rumah.

## Data Understanding
Dataset California Housing Prices dari Kaggle berisi informasi mengenai harga rumah di wilayah California dengan berbagai fitur demografis dan karakteristik rumah. Berikut adalah gambaran umum mengenai isi dataset ini:

1. Ukuran Dataset
   - Jumlah sampel: Lebih dari 20.000 data.
   - Target variabel: median_house_value, yang merupakan harga median dari rumah di wilayah tertentu (dalam dolar AS).

2. Karakteristik Data
   - Tipe Data
  
      Sebagian besar fitur bersifat kuantitatif, dengan nilai-nilai numerik seperti jumlah kamar dan usia rumah.

   - Distribusi Geografi
  
      Data memiliki variasi lokasi yang cukup baik di seluruh California, memungkinkan model untuk menangkap perbedaan harga berdasarkan letak geografis.

   - Skala Pendapatan
      
      Median_income dalam dataset ini menjadi salah satu indikator penting karena berkorelasi kuat dengan harga rumah.

1. link
   - https://www.kaggle.com/datasets/camnugent/california-housing-prices

   - ### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
      Dataset ini mencakup beberapa fitur kuantitatif yang relevan untuk prediksi harga rumah, di antaranya:
      - longitude dan latitude: Lokasi geografis dari daerah di mana rumah tersebut berada.
      - housing_median_age: Umur median rumah di suatu area.
      - total_rooms: Total jumlah kamar di area tersebut.
      - total_bedrooms: Total jumlah kamar tidur.
      - population: Jumlah populasi di area tersebut.
      - households: Jumlah rumah tangga di area tersebut.
      - median_income: Pendapatan median dari penduduk di area tersebut (dalam puluhan ribu dolar AS).
      - median_house_value: Nilai median dari harga rumah di area tersebut (ini merupakan target prediksi).
      - oceanProximity: Lokasi rumah terhadap laut/samudra.

2. Missing value
   - Terdapat data yang kosong pada kolom `total_bedrooms` sebanyak 207 data

## Data Preparation
  Saya melakukan beberapa cara agar data menjadi lebih baik sebelum melakukan proses training.
- Feature Selection
  
  Melakukan penghapusan terhadap 207 data yang missing value pada kolom `total_bedrooms` menggunakan ```.dropna(inplace=True)``` pada data tersebut, karena dapat mempengaruhi kualitas data. 
- One-Hot Encoding
  
  Karena terdapat data yang bertype objek, maka dilakukan proses merubah data tersebut menjadi boolean menggunakan `onehot encoding`. Tindakan ini dilakukan agar mempermudah proses training data.
- Split Data
  
  Kemudian melakukan split data 80% data training dan 20% data test.

## Modeling
Saya menggunakan metode LSTM dan GRU sebagai pembanding. Parameter yang saya digunakan sebagai berikut:

|parameter|nilai|
|---------|------|
|epochs|20|
|batch_size|32|
|optimizer|adam|
|loss|mse|
|verbose|1|

### LSTM

- Penjelasan

   LSTM (Long Short-Term Memory) adalah jaringan saraf tiruan yang dirancang untuk memproses data sekuensial dengan menangkap pola jangka panjang. LSTM mengatasi kelemahan utama RNN tradisional, yaitu *vanishing gradient*, melalui penggunaan tiga jenis *gate* (input, forget, dan output gate) yang memungkinkan kontrol atas informasi mana yang harus disimpan, dilupakan, atau digunakan. Hal ini membuat LSTM sangat cocok untuk tugas-tugas seperti prediksi time-series, pemrosesan bahasa alami, dan pengenalan suara [(Hochreiter & Schmidhuber, 1997; Greff et al., 2017)](https://doi.org/10.1162/neco.1997.9.8.1735).

- Keunggulan

   Keunggulan utama LSTM adalah fleksibilitasnya dalam menangkap pola kompleks dalam data yang memiliki hubungan temporal atau dependensi antar fitur. Mekanisme memorinya memungkinkan model ini mempertahankan informasi penting dari konteks sebelumnya, sehingga efektif dalam aplikasi seperti peramalan harga saham, analisis sentimen, dan prediksi harga rumah. Namun, dibandingkan dengan model lain seperti GRU, LSTM lebih kompleks secara komputasi dan memerlukan dataset besar untuk mencapai performa optimal [(Gers et al., 2000; Siami-Namini et al., 2019)](https://doi.org/10.1162/089976600300015015).

- Kekurangan

   Kekurangan LSTM mencakup waktu pelatihan yang lebih lama dan sensitivitas terhadap parameter model. Selain itu, jika dataset kecil atau tidak representatif, LSTM rentan terhadap overfitting. Dalam kasus data non-time series atau kebutuhan prediksi cepat, algoritma seperti Random Forest atau XGBoost dapat menjadi alternatif yang lebih praktis [(Greff et al., 2017; Siami-Namini et al., 2019)](https://doi.org/10.1109/TNNLS.2016.2582924).

- Alur
  - Dalam proses ini, data terlebih dahulu di-*scaling* menggunakan `MinMaxScaler` untuk memastikan semua fitur memiliki nilai dalam rentang tertentu, sehingga mempermudah proses pembelajaran model. 
  - Data pelatihan (`X_train` dan `y_train`) di-*fit_transform*, sedangkan data pengujian (`X_test` dan `y_test`) hanya di-*transform* agar konsisten dengan skala data pelatihan. 
  - Selanjutnya, data di-*reshape* ke format 3D `(samples, timesteps, features)` agar kompatibel dengan input model `LSTM (Long Short-Term Memory)`, yang dirancang untuk data berurutan. 
  - Model LSTM dibuat dengan lapisan tersembunyi yang terdiri dari 50 unit, fungsi aktivasi `relu`, dan lapisan keluaran `Dense` dengan fungsi aktivasi `linear` untuk prediksi nilai kontinu. 
  - Model dikompilasi menggunakan optimizer `adam` dan metrik evaluasi  `Mean Squared Error (MSE)`. 
  - Setelah pelatihan model pada data pelatihan dengan beberapa `epochs` dan ukuran `batch` yang ditentukan, prediksi dilakukan pada data pengujian. 
  - Akhirnya, model dievaluasi menggunakan metrik `MSE` untuk mengukur tingkat kesalahan prediksi terhadap data yang telah di-*scaled*.


### GRU

- Penjelasan
   
   GRU (Gated Recurrent Unit) adalah jenis jaringan saraf tiruan yang termasuk dalam keluarga Recurrent Neural Network (RNN). GRU diperkenalkan oleh [Cho et al. (2014)](https://aclanthology.org/D14-1179) sebagai alternatif yang lebih sederhana daripada LSTM. Sama seperti LSTM, GRU dirancang untuk mengatasi masalah *vanishing gradient* yang sering terjadi pada RNN, dengan menggunakan dua jenis gate, yaitu *update gate* dan *reset gate*, untuk mengontrol informasi yang diteruskan melalui jaringan. Perbedaan utama antara GRU dan LSTM adalah bahwa GRU tidak memiliki *forget gate* dan lebih sederhana dalam struktur, dengan hanya dua gate dibandingkan dengan tiga pada LSTM.

- Keunggulan

   Keunggulan utama GRU adalah kesederhanaannya. Dibandingkan dengan LSTM, GRU memiliki lebih sedikit parameter dan lebih cepat dalam pelatihan, sehingga lebih efisien dalam hal waktu komputasi dan memori, terutama ketika berhadapan dengan dataset yang besar atau ketika sumber daya komputasi terbatas [Cho et al. (2014)](https://aclanthology.org/D14-1179); [Chung et al., 2014](https://papers.nips.cc/paper/2014/hash/5f8f5b1684b5ad5b5444e6f3d2cc9c72-Abstract.html). Selain itu, GRU sering kali memberikan performa yang sebanding dengan LSTM dalam banyak aplikasi, tetapi dengan waktu pelatihan yang lebih cepat.

- Kekurangan

   Namun, GRU juga memiliki kekurangan. Meskipun lebih efisien, GRU mungkin tidak selalu dapat menangkap hubungan jangka panjang dalam data dengan kualitas yang sama seperti LSTM, terutama pada tugas-tugas yang membutuhkan ketepatan memori yang lebih baik. Selain itu, dalam beberapa kasus, GRU mungkin tidak dapat menggantikan LSTM pada masalah yang sangat kompleks, seperti peramalan time series yang sangat bergantung pada konteks jangka panjang ([Chung et al., 2014](https://papers.nips.cc/paper/2014/hash/5f8f5b1684b5ad5b5444e6f3d2cc9c72-Abstract.html); [Li et al., 2019](https://doi.org/10.1155/2019/8326093)).

- Alur
  
  - Dalam proses ini, data di-*scaling* menggunakan `MinMaxScaler` untuk memastikan semua fitur memiliki nilai dalam rentang tertentu, mempermudah pembelajaran model. 
  - Data pelatihan (`X_train` dan `y_train`) di-*fit_transform*, sementara data pengujian (`X_test` dan `y_test`) di-*transform* untuk konsistensi skala. 
  - Setelah itu, data di-*reshape* ke format 3D `(samples, timesteps, features)` agar kompatibel dengan model `GRU (Gated Recurrent Unit)`, yang dirancang untuk menangani data berurutan. 
  - Model GRU dibangun dengan satu lapisan tersembunyi berisi 50 unit, menggunakan fungsi aktivasi `relu` dan lapisan keluaran `Dense` untuk prediksi nilai kontinu. 
  - Model dikompilasi menggunakan optimizer `adam` dan fungsi kerugian berupa rata-rata kuadrat kesalahan atau `Mean Squared Error (MSE)`. 
  - Setelah proses pelatihan pada data pelatihan dengan beberapa `epochs` dan ukuran `batch` yang telah ditentukan, model digunakan untuk memprediksi data pengujian. 
  - Kinerja model dievaluasi menggunakan metrik `MSE` untuk mengukur seberapa baik prediksi dibandingkan dengan data yang telah di-*scaled*.




## Evaluation

### Perbandingan Nilai MSE
![alt text](image/mse%20LSTM%20vs%20GRU.png)

Grafik ini membandingkan performa dua model, **LSTM (Long Short-Term Memory)** dan **GRU (Gated Recurrent Unit)**, dalam tugas prediksi harga rumah berdasarkan nilai **Mean Squared Error (MSE)**. Sumbu vertikal menunjukkan MSE, yaitu metrik yang mengukur seberapa jauh prediksi model dari nilai aktual. Semakin kecil nilai MSE, semakin baik performa model. Berdasarkan grafik, model GRU (batang merah) memiliki MSE sebesar **0.0158**, sedikit lebih rendah dibandingkan model LSTM (batang biru) yang memiliki MSE sebesar **0.0163**. Hal ini menunjukkan bahwa GRU lebih akurat dalam memprediksi harga rumah pada dataset yang digunakan. Perbedaan ini, meskipun kecil, dapat menjadi bahan pertimbangan dalam memilih model yang lebih efisien untuk proyek prediksi harga rumah ke depannya.

### Perbandingan Hasil Training Loss
![alt text](image/training%20loss%20LSTM%20vs%20GRU.png)

Grafik ini menunjukkan perbandingan **training loss** antara model **LSTM** dan **GRU** selama proses pelatihan pada tugas prediksi harga rumah. Sumbu horizontal merepresentasikan jumlah **epoch** (jumlah iterasi pelatihan), sedangkan sumbu vertikal menunjukkan nilai **loss**, yang menggambarkan seberapa besar kesalahan model selama pelatihan. Pada awal pelatihan (epoch 0-2), kedua model mengalami penurunan loss yang signifikan. Namun, setelah beberapa epoch, penurunan loss menjadi lebih stabil dan perlahan mendekati nilai minimum.

Model GRU (garis oranye) secara konsisten memiliki nilai loss yang sedikit lebih rendah dibandingkan LSTM (garis biru), menunjukkan bahwa GRU belajar lebih cepat dan lebih efisien dalam mengurangi kesalahan selama pelatihan. Pada akhir pelatihan, kedua model hampir konvergen dengan nilai loss sekitar **0.015**, namun GRU tetap unggul dalam hal performa pelatihan. Perbandingan ini mengindikasikan bahwa GRU dapat menjadi pilihan yang lebih baik untuk tugas prediksi harga rumah, terutama ketika efisiensi pelatihan menjadi pertimbangan.

Kedua metode tersebut sudah menjawab Problem Statements dan mencapai goals yang dituju yaitu berhasil mengambangkan model prediktif untuk pola kompleks dan hubungan non-linear dalam data harga rumah. Kedua metode tersebut menampilkan hasil nilai error yang terbilang cukup kecil untuk masalah sebesar prediksi harga rumah dengan menggunakan banyak fitur. Untuk selanjutnya bisa melakukan penelitian dengan metode berbeda dengan metode yang telah digunakan untuk menambah referensi untuk prediksi harga rumah.

## Conclusion

Proyek ini bertujuan untuk memprediksi harga rumah menggunakan model deep learning berbasis **LSTM** dan **GRU**, dengan data *California Housing Prices* sebagai acuan. Analisis menunjukkan bahwa GRU memiliki keunggulan dalam hal efisiensi pelatihan dan akurasi, dengan nilai **Mean Squared Error (MSE)** lebih rendah dibandingkan LSTM. GRU juga menunjukkan performa yang lebih baik dalam mengurangi training loss secara konsisten. Hasil ini menyoroti potensi GRU sebagai model yang lebih ringan dan efisien dalam menangkap pola kompleks untuk prediksi harga rumah, terutama di pasar dinamis seperti California. Temuan ini memberikan wawasan yang dapat digunakan untuk menganalisis dinamika harga properti di Indonesia, meskipun konteks pasar berbeda, dan dapat menjadi referensi untuk pengambilan keputusan investasi di sektor properti.

## Referensi
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. *Neural Computation, 12*(10), 2451-2471. https://doi.org/10.1162/089976600300015015  
- Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A search space odyssey. *IEEE Transactions on Neural Networks and Learning Systems, 28*(10), 2222-2232. https://doi.org/10.1109/TNNLS.2016.2582924  
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735  
- Siami-Namini, S., Tavakoli, N., & Namin, A. S. (2019). The performance of LSTM and BiLSTM in forecasting time series. *2019 IEEE International Conference on Big Data (Big Data)*, 3285–3292. https://doi.org/10.1109/BigData47090.2019.9005997  
- Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *Proceedings of EMNLP 2014*, 1724-1734. https://aclanthology.org/D14-1179
- Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *Proceedings of NIPS 2014*, 1018-1026. https://papers.nips.cc/paper/2014/hash/5f8f5b1684b5ad5b5444e6f3d2cc9c72-Abstract.html
- Li, Y., Liang, H., & Zheng, Y. (2019). A comprehensive review on GRU and LSTM: Applications and challenges. *Computational Intelligence and Neuroscience*, 2019, 1-12. https://doi.org/10.1155/2019/8326093
