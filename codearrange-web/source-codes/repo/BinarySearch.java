// Description: Kode di bawah ini mencari nilai target dalam array terurut menggunakan Binary Search. Susunlah Output/Keluaran dari kodenya. 
// Cognitive Load Rating: 6
// bagus utk tipe 2, output
// Kode
```public class BinarySearch { \n    public static void main(String[] args) { \n        int[] arr = {2, 3, 4, 7, 9, 10, 15, 20, 40}; \n        int target = 10; \n        int l = 0, r = arr.length - 1; \n        while (l <= r) { \n            int m = l + (r - l) / 2; \n            System.out.println(arr[m]); \n            if (arr[m] == target) { \n                System.out.println(\"Found\"); \n                break; \n            } \n            if (arr[m] < target) l = m + 1; \n            else r = m - 1; \n        } \n    } \n}```


// Output
9
15
10
Found
2
3
4
7
20
40