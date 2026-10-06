// Description: Transpose / tukar baris dan kolom sebuah Matriks.
// Cognitive Load Rating: 7

public class MatrixTranspose {
    public static void main(String[] args) {
        int[][] original = { {1, 2, 3}, {4, 5, 6} };
        int rows = original.length;
        int cols = original[0].length;
        int[][] transposed = new int[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = original[i][j];
            }
        }
    }
}
