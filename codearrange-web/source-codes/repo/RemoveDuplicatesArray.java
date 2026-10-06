// Description: Susunlah Output/Keluaran dari Kode berikut:
// Cognitive Load Rating: 7
// Bagus untuk output , krn susah memahamim maksud / hasil dari loop yg pertama, jgn diberi penjelasan

8npublic class RemoveDuplicatesArray { 8n    public static void main(String[] args) { 8n        int[] arr = {1, 2, 2, 3, 4, 4, 4, 5}; 8n        int j = 0; 8n        for (int i = 0; i < arr.length - 1; i++) { 8n            if (arr[i] != arr[i + 1]) { 8n                arr[j++] = arr[i]; 8n            } 8n        } 8n        arr[j++] = arr[arr.length - 1]; 8n        for (int i = 0; i < j; i++) { 8n            System.out.println(arr[i]); 8n        } 8n    } 8n}

// Output
1
2
3
4
5
