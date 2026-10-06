// Description: Calculates the next generation of Conway's Game of Life on a 2D grid.
// Cognitive Load Rating: 9

public class GameOfLife {
    public void nextGen(int[][] board) {
        int m = board.length, n = board[0].length;
        int[][] next = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int lives = 0;
                for (int x = Math.max(0, i-1); x <= Math.min(m-1, i+1); x++) {
                    for (int y = Math.max(0, j-1); y <= Math.min(n-1, j+1); y++) {
                        lives += board[x][y];
                    }
                }
                lives -= board[i][j];
                if (board[i][j] == 1 && (lives == 2 || lives == 3)) next[i][j] = 1;
                else if (board[i][j] == 0 && lives == 3) next[i][j] = 1;
            }
        }
    }
}
