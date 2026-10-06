// Description: Traverses a 2D matrix in a spiral pattern tracking 4 boundary pointers.
// Cognitive Load Rating: 9
// Bagus untuk output

public class SpiralMatrix {
    public static void main(String[] args) {
        int[][] mat = {{1,2,3},{4,5,6},{7,8,9}};
        int top = 0, bottom = 2, left = 0, right = 2;
        while (top <= bottom && left <= right) {
            for (int i = left; i <= right; i++) System.out.print(mat[top][i] + " ");
            top++;
            for (int i = top; i <= bottom; i++) System.out.print(mat[i][right] + " ");
            right--;
            if (top <= bottom) {
                for (int i = right; i >= left; i--) System.out.print(mat[bottom][i] + " ");
                bottom--;
            }
            if (left <= right) {
                for (int i = bottom; i >= top; i--) System.out.print(mat[i][left] + " ");
                left++;
            }
        }
    }
}
