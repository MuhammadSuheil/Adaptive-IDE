# Catatan Tambahan & Bahan Diskusi: Eye Tracking Module

Berdasarkan *draft planning* (`draft-1.md`) yang telah disusun, arsitektur dan pendekatannya sudah sangat baik. Berikut adalah beberapa poin teknis tambahan yang perlu didiskusikan lebih lanjut bersama tim, terutama karena sistem ini menggabungkan dua modul pengumpulan data yang berbeda (Eye Tracking dan Heart Rate).

## 1. Pentingnya Sinkronisasi Waktu (*Time Sync*)
Karena sistem ini melakukan **Fusi Sensor Multimodal**, pastikan setiap paket data (misalnya dalam format JSON) yang dikirim dari aplikasi Python (Eye Tracking) dan aplikasi Heart Rate menyertakan **Timestamp (dalam milidetik)** yang tersinkronisasi. Jika waktu perekamannya tidak sinkron, data HR dan gerakan mata akan sangat sulit dicocokkan ketika dianalisis oleh algoritma *Machine Learning* untuk menyimpulkan kondisi kognitif *developer*.

## 2. Hit Detection pada Sumbu X (Horizontal)
Pada *Phase 5: Hit Detection* di *draft*, rumus yang dicantumkan saat ini masih berfokus pada Sumbu Y (mencari baris kode ke-berapa). Mungkin perlu dipertimbangkan untuk memperhitungkan rentang Sumbu X (horizontal) juga. Hal ini penting untuk mendeteksi apakah *developer* benar-benar sedang membaca isi baris kode tersebut, ataukah matanya sedang melirik ke arah *Explorer/Sidebar* di sebelah kiri, ataupun menatap *Minimap* di sebelah kanan.

## 3. Jeda Respons Fisiologis (HR Delay)
Dalam mendeteksi *User State* seperti **Confused** (Dwell lama di area error + HR tinggi), perlu diingat bahwa perubahan secara biologis seperti detak jantung (HR/HRV) akibat stres kognitif biasanya mengalami *delay* atau jeda waktu selama beberapa detik setelah *developer* melihat sesuatu yang membingungkan. Oleh karena itu, algoritma atau *State Detection* harus memperhitungkan sebuah jendela waktu (*time window*), bukan sekadar mencocokkan data detak jantung dan pergerakan mata pada detik atau milidetik yang persis sama.
