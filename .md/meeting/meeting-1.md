Kita perlu melakukan pengumpulan data / membuat dataset sendiri (primary)  
karena mau multimodal fusi sensor

data dari heart rate dan dari eye movement harus sinkron

1\. Data Heart Rate

Perlu sensor  
Data yang diperlukan adalah Heart Rate Variability (HRV), bukan Beat Per Minute (BPM)  
Yang sering kita liat itu adalah BPM

Menambah kompleksitas karena:  
\- tidak semua sensor punya data HRV, kalo BPM pasti punya  
\- ada 2 alternatif sensor:  
  \* ECG: sinyal listrik, tapi harus dipasang di data (chest), lebih akurat tapi tidak convenience, bisa jadi ada efek ke beban kognitif, harganya lebih mahal  
  \* PPG: sinyal optik, bisa dipasang di pergelangan atau di lengan atas, tidak seakurat ECG tapi nyaman, tidak menganggu, harga lebih murah

Di penelitian ini kita akan menggunakan PPG  
ada 2 jenis device (krn kita tidak bikin alat/hardware sendiri) yang tersedia (ready shelf)  
\- Smart Watch (bukan smart band, smart band tidak punya data HRV)  
  Masalahnya di harga (krn kita tidak boleh beli alat) 3 juta ke atas (samsung smart watch)  
 Beli seken, banyak sekali palsuan  
\- HRM Arm Band  
  spesifik sensor Heart Rate aja, di-pair dan broadcast data melalui BLE (Bluetooth Low Energy)  
  tapi tidak semua sensor broadcast HRV, harus pilih2  
  Polar H10, mahal  
  Coospo HW9 , merk cina, budget version

TODO: membuat aplikasi yang bisa mengambil data HRV dari Coospo HW9, katanya menggunakan standar protokol BLE  
\- aplikasi apa?  
  \- Android → App Desktop

\- karena objek penelitian menggunakan IDE di depan laptop, dan juga akan digabung dengan data Eye Movement

2\. Data Eye Movement

Saya bilang, ada software-nya seharga 30 dollar. Software ini untuk main game, artinya output-nya adalah gerakan2 joystick/controller. Bukan yang kita perlukan.

Sudah banyak open-source library, baik pake Python, atau Javascript (menggunakan Machine Learning)

Artinya kita harus develop software-nya juga untuk mengambil data Eye Movement

Menggunakan hardware berupa Webcam

spec webcam yang direkomendasikan:

60 fps, 1080p

Fantech C50 \-\> 2K @30fps atau 1080p @60fps 

sambil menunggu webcam ini, bisa sementara menggunakan webcam di laptop biasa

3\. Membuat skenario testing  
\- situasinya: sedang menggunakan IDE  
\- kita ambil HRV dan Eye Movement orang tsb

Perlu labelling atau ground truth-nya tentang kondisi kognitif pengembang pada setiap skenario  
→ perlu support software-nya utk membantu menjalankan skenario2-nya

TODO:  
\- di weekend ini, pelajari tentang no 1 dan 2, khususnya untuk development di Android ataupun Desktop

\- minggu depan kita bagi2 tugas, diputuskan siapa yang fokus di no 1, siapa yang fokus di no 2

