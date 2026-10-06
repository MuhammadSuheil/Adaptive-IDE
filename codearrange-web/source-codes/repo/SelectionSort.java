// Description: Sorts an array using the Selection Sort algorithm.
// Cryptic Desc: Susunlah Output/Keluaran dari Kode berikut:
// Cognitive Load Rating: 6
// Bagus untuk output

```public class SelectionSort { \n    public static void main(String[] args) { \n        int[] arr = {64, 25, 12, 22, 11}; \n        for (int i = 0; i < arr.length - 1; i++) { \n            int minIdx = i; \n            for (int j = i + 1; j < arr.length; j++) { \n                if (arr[j] < arr[minIdx]) { \n                    minIdx = j; \n                } \n            } \n            int temp = arr[minIdx]; \n            arr[minIdx] = arr[i]; \n            arr[i] = temp; \n        } \n    } \n}```

// Output
25
12
11
12
22
64
