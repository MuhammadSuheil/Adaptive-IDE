# Evaluasi & Penentuan Rating Tingkat Kesulitan Soal Code Arrange

## Rubrik Skala Tingkat Kesulitan (1 - 10)

| Rentang Rating | Kategori | Karakteristik Masalah (Perspektif Mahasiswa Semester 3) |
|:---:|:---:|---|
| **1 - 2** | **Sangat Mudah - Mudah** | Alur sekuensial lurus, I/O dasar, deklarasi variabel sederhana, percabangan tunggal (`if-else`), perulangan `for` akumulasi standar. |
| **3 - 4** | **Cukup Mudah - Sedang Bawah** | Perulangan dengan kondisi terminasi manual (`while`), manipulasi string/karakter dasar, pencarian sekuensial (*linear search*), duplikasi array 1D. |
| **5 - 6** | **Sedang - Menengah** | Perulangan bersarang (*nested loops*), manipulasi matriks/array 2D, konsep penukaran (*swapping*) pada Bubble Sort, algoritma *two-pointers* dasar, *tracing* kalkulasi *midpoint* pada Binary Search. |
| **7 - 8** | **Sulit - Sangat Menantang** | Urutan pergeseran variabel state yang rentan tertukar (misal deret Fibonacci), kondisi penanganan kasus batas (*edge cases* seperti bilangan prima), logika *multi-pass* frekuensi karakter (Anagram, First Non-Repeating Char), teknik penggabungan *three-pointers* (*Merge Sorted Arrays*), *tracing* mutasi bersyarat dalam *Selection Sort*. |
| **9 - 10** | **Ekstrem / Ahli** | Algoritma rekursif kompleks tingkat lanjut, graf, *dynamic programming*, atau struktur data lanjutan non-linear (tidak ada pada set soal dasar ini). |

---

## Ringkasan Rekapitulasi Rating Final (34 Soal)

Berikut rekapitulasi lengkap 34 soal yang telah dipetakan secara sistematis:

| No | Exam Set | ID Soal | Judul Soal | Tipe Puzzle | Jml Baris | Waktu Rata-rata Tim | Rentang Rating Tim (Min - Max) | Rata-rata Rating Tim | **Rating Final Rekomendasi** | Kategori Kesulitan |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `exam-set-a` | `hello_world` | 1. Hello, World! | Parsons Code | 5 | 11.2s | 1 - 1 | 1.00 | **1** | Sangat Mudah |
| 2 | `exam-set-a` | `sum_two_numbers` | 2. Penjumlahan Dua Angka | Parsons Code | 8 | 43.7s | 1 - 1 | 1.00 | **1** | Sangat Mudah |
| 3 | `exam-set-a` | `for_loop_sum` | 3. Penjumlahan 1-10 dg Loop | Parsons Code | 9 | 93.9s | 1 - 2 | 1.40 | **2** | Mudah |
| 4 | `exam-set-b` | `sum_two_numbers` | 2. Penjumlahan Dua Angka | Parsons Code | 8 | 30.0s | 1 - 1 | 1.00 | **1** | Sangat Mudah |
| 5 | `exam-set-b` | `for_loop_sum` | 3. Penjumlahan 1-10 dg Loop | Parsons Code | 9 | 81.2s | 1 - 2 | 1.20 | **2** | Mudah |
| 6 | `exam-set-b` | `fix_even_sum` | 4. Perbaiki Penjumlahan Genap | Parsons Code | 11 | 197.5s | 1 - 3 | 2.20 | **3** | Cukup Mudah |
| 7 | `exam-set-c` | `1_variables` | Variables | Parsons Code | 7 | 26.5s | 1 - 2 | 1.40 | **1** | Sangat Mudah |
| 8 | `exam-set-c` | `2_even_odd` | Even Odd | Parsons Code | 10 | 105.5s | 1 - 2 | 1.40 | **2** | Mudah |
| 9 | `exam-set-c` | `2_simple_ifelse` | Simple If Else | Parsons Code | 10 | 154.6s | 1 - 2 | 1.40 | **2** | Mudah |
| 10 | `exam-set-c` | `2_string_concat` | String Concat | Parsons Code | 8 | 207.3s | 1 - 1 | 1.00 | **1** | Sangat Mudah |
| 11 | `exam-set-c` | `3_array_sum` | Array Sum | Parsons Code | 10 | 259.8s | 1 - 3 | 1.60 | **2** | Mudah |
| 12 | `exam-set-c` | `3_basic_for_loop` | Print Numbers | Parsons Code | 7 | 292.7s | 1 - 3 | 1.40 | **2** | Mudah |
| 13 | `exam-set-c` | `3_basic_while_loop` | Print Numbers Reverse | Parsons Code | 9 | 385.4s | 1 - 4 | 2.50 | **3** | Cukup Mudah |
| 14 | `exam-set-c` | `3_find_max` | Find Max | Parsons Code | 12 | 472.4s | 1 - 5 | 2.80 | **3** | Cukup Mudah |
| 15 | `exam-set-c` | `4_count_vowels` | Count Vowels | Parsons Code | 13 | 665.1s | 3 - 7 | 4.40 | **4** | Sedang |
| 16 | `exam-set-c` | `4_factorial_iterative` | Factorial Iterative | Parsons Code | 10 | 734.7s | 1 - 4 | 2.60 | **3** | Cukup Mudah |
| 17 | `exam-set-c` | `4_fibonacci_iterative` | Fibonacci Iterative | Parsons Code | 12 | 1488.8s | 5 - 9 | 7.40 | **7** | Sulit |
| 18 | `exam-set-c` | `4_linear_search` | Linear Search | Parsons Code | 14 | 1652.0s | 3 - 9 | 4.60 | **4** | Sedang |
| 19 | `exam-set-c` | `4_string_reverse` | String Reverse | Parsons Code | 10 | 1697.3s | 2 - 8 | 3.60 | **4** | Sedang |
| 20 | `exam-set-c` | `array_copy_manual` | Array Copy Manual | Parsons Code | 9 | 1762.1s | 2 - 7 | 3.40 | **3** | Cukup Mudah |
| 21 | `exam-set-c` | `5_matrix_addition` | Matrix Addition | Parsons Code | 12 | 1847.3s | 3 - 8 | 4.20 | **5** | Sedang |
| 22 | `exam-set-c` | `5_palindrome_check` | Palindrome Check | Parsons Code | 17 | 2155.7s | 4 - 9 | 5.80 | **6** | Menengah |
| 23 | `exam-set-c` | `5_prime_check` | Prime Check | Parsons Code | 16 | 2724.6s | 5 - 9 | 7.00 | **7** | Sulit |
| 24 | `exam-set-c` | `6_bubble_sort` | Bubble Sort | Parsons Code | 14 | 2976.9s | 3 - 8 | 5.60 | **6** | Menengah |
| 25 | `exam-set-c` | `anagram_check` | Anagram Check | Parsons Code | 16 | 3503.5s | 7 - 10 | 8.00 | **8** | Sangat Menantang |
| 26 | `exam-set-c` | `7_caesar_cipher` | Caesar Cipher | Parsons Code | 12 | 3648.4s | 3 - 7 | 5.20 | **6** | Menengah |
| 27 | `exam-set-c` | `7_matrix_transpose` | Matrix Transpose | Parsons Code | 13 | 3759.9s | 3 - 7 | 4.60 | **5** | Sedang |
| 28 | `exam-set-c` | `8_first_non_repeating_char` | First Non Repeating Char | Parsons Code | 18 | 4151.9s | 4 - 10 | 7.40 | **8** | Sangat Menantang |
| 29 | `exam-set-c` | `8_merge_sorted_arrays` | Merge Sorted Arrays | Parsons Code | 13 | 4498.2s | 7 - 10 | 8.40 | **8** | Sangat Menantang |
| 30 | `exam-set-d` | `1_hello_world` | Cetak String | Trace Output | 2 | 16.5s | 1 - 1 | 1.00 | **1** | Sangat Mudah |
| 31 | `exam-set-d` | `4_print_2darray` | Cetak Array 2D | Trace Output | 3 | 115.8s | 1 - 3 | 2.00 | **2** | Mudah |
| 32 | `exam-set-d` | `6_binary_search` | Binary Search | Trace Output | 4 | 314.4s | 3 - 10 | 5.40 | **6** | Menengah |
| 33 | `exam-set-d` | `6_selection_sort` | Manipulasi Array 1 | Trace Output | 5 | 544.5s | 4 - 10 | 6.40 | **7** | Sulit |
| 34 | `exam-set-d` | `7_remove_duplicates` | Manipulasi Array 2 | Trace Output | 5 | 646.9s | 1 - 10 | 4.80 | **6** | Menengah |

---

## Distribusi Tingkat Kesulitan Soal

```
[Level 1]  ██████ (6 Soal)   - Trivial (Hello World, Aritmatika Sekuensial, String Concat)
[Level 2]  ███████ (7 Soal)  - Mudah (Loop For Dasar, If-Else, Array 1D Sum, Tracing 2D)
[Level 3]  █████ (5 Soal)    - Cukup Mudah (While Loop Reverse, Find Max, Factorial, Copy Array)
[Level 4]  ███ (3 Soal)      - Sedang Bawah (Count Vowels, Linear Search, String Reverse)
[Level 5]  ██ (2 Soal)       - Sedang (Matrix Addition, Matrix Transpose)
[Level 6]  █████ (5 Soal)    - Menengah (Palindrome, Bubble Sort, Caesar, Tracing Binary Search, Tracing Duplicates)
[Level 7]  ███ (3 Soal)      - Sulit (Fibonacci Update State, Prime Check, Tracing Selection Sort)
[Level 8]  ███ (3 Soal)      - Sangat Menantang (Anagram Check, First Non-Repeating, Merge Sorted Arrays)
```

- **Proporsi Soal Mudah (Rating 1 - 3)**: **18 Soal (53%)**
- **Proporsi Soal Menengah (Rating 4 - 6)**: **10 Soal (29%)**
- **Proporsi Soal Sukar/Tantangan (Rating 7 - 8)**: **6 Soal (18%)**
