// Description: Generates Pascal's Triangle using a nested loop mathematical approach.
// Cognitive Load Rating: 8

public class PascalTriangle {
    public static void main(String[] args) {
        int n = 5;
        for (int line = 1; line <= n; line++) {
            int C = 1;
            for (int i = 1; i <= line; i++) {
                System.out.print(C + " ");
                C = C * (line - i) / i;
            }
            System.out.println();
        }
    }
}
