// Description: Solves a 2D grid maze using a recursive Depth-First Search approach.
// Cognitive Load Rating: 9

public class MazeSolverDFS {
    public boolean solve(int[][] maze, int x, int y, int[][] sol) {
        int n = maze.length;
        if (x == n - 1 && y == n - 1 && maze[x][y] == 1) {
            sol[x][y] = 1;
            return true;
        }
        if (x >= 0 && x < n && y >= 0 && y < n && maze[x][y] == 1) {
            sol[x][y] = 1;
            if (solve(maze, x + 1, y, sol)) return true;
            if (solve(maze, x, y + 1, sol)) return true;
            sol[x][y] = 0;
        }
        return false;
    }
}
