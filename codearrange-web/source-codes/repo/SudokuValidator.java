// Description: Validates a 9x9 Sudoku grid using coordinate tracking and hash-based logic.
// Cognitive Load Rating: 9

public class SudokuValidator {
    public static boolean isValid(int[][] board) {
        for (int i = 0; i < 9; i++) {
            int[] row = new int[10];
            int[] col = new int[10];
            int[] box = new int[10];
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != 0 && ++row[board[i][j]] > 1) return false;
                if (board[j][i] != 0 && ++col[board[j][i]] > 1) return false;
                int rowIdx = 3 * (i / 3) + j / 3;
                int colIdx = 3 * (i % 3) + j % 3;
                if (board[rowIdx][colIdx] != 0 && ++box[board[rowIdx][colIdx]] > 1) return false;
            }
        }
        return true;
    }
}
