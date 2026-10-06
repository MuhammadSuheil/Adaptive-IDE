// Description: Evaluates a 2D array to check if there is a Tic-Tac-Toe winner.
// Cognitive Load Rating: 8

public class TicTacToeCheck {
    public static void main(String[] args) {
        char[][] board = {
            {'X', 'O', 'X'},
            {'O', 'X', 'O'},
            {'O', 'O', 'X'}
        };
        char winner = '-';
        for (int i = 0; i < 3; i++) {
            if (board[i][0] == board[i][1] && board[i][1] == board[i][2]) winner = board[i][0];
            if (board[0][i] == board[1][i] && board[1][i] == board[2][i]) winner = board[0][i];
        }
        if (board[0][0] == board[1][1] && board[1][1] == board[2][2]) winner = board[0][0];
        if (board[0][2] == board[1][1] && board[1][1] == board[2][0]) winner = board[0][2];
        System.out.println("Winner: " + winner);
    }
}
