// Description: Calculates Least Common Multiple using a continuous loop.
// Cognitive Load Rating: 6

public class LcmCalc {
    public static void main(String[] args) {
        int n1 = 72, n2 = 120;
        int lcm = (n1 > n2) ? n1 : n2;
        while (true) {
            if (lcm % n1 == 0 && lcm % n2 == 0) {
                System.out.println("LCM: " + lcm);
                break;
            }
            lcm++;
        }
    }
}
