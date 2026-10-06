// Description: Calculates factorial using method recursion.
// Cognitive Load Rating: 5

public class FactorialRecursive {
    static int fact(int n) {
        if (n == 0) return 1;
        return n * fact(n - 1);
    }
    public static void main(String[] args) {
        System.out.println(fact(5));
    }
}
