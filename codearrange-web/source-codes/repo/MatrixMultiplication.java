// Description: Multiplies two matrices requiring triple nested loops to compute dot products.
// Cognitive Load Rating: 8

public class MatrixMultiplication {
    public static void main(String[] args) {
        int[][] a = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
        int[][] b = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
        int[][] c = new int[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                c[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
}
