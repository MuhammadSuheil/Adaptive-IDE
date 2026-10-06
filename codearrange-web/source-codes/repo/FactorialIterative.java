// Description: Hitung nilai Faktorial dari n menggunakan loop (non-recursive).
// Cognitive Load Rating: 4

public class FactorialIterative {
    public static void main(String[] args) {
        int n = 5;
        long fact = 1;
        for (int i = 1; i <= n; i++) {
            fact = fact * i;
        }
        System.out.println("Factorial: " + fact);
    }
}
