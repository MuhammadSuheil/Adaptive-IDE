// Description: Calculates Greatest Common Divisor using Euclidean algorithm.
// Cognitive Load Rating: 5

public class GcdCalc {
    public static void main(String[] args) {
        int a = 60, b = 48;
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        System.out.println("GCD: " + a);
    }
}
