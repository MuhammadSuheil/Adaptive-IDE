// Description: Solves the N-Queens problem on a chess board using recursive backtracking.
// Cognitive Load Rating: 10

public class NQueensBacktracking {
    boolean isSafe(int board[][], int row, int col) {
        for (int i = 0; i < col; i++) if (board[row][i] == 1) return false;
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) if (board[i][j] == 1) return false;
        for (int i = row, j = col; j >= 0 && i < board.length; i++, j--) if (board[i][j] == 1) return false;
        return true;
    }
    boolean solveNQUtil(int board[][], int col) {
        if (col >= board.length) return true;
        for (int i = 0; i < board.length; i++) {
            if (isSafe(board, i, col)) {
                board[i][col] = 1;
                if (solveNQUtil(board, col + 1)) return true;
                board[i][col] = 0;
            }
        }
        return false;
    }
}
