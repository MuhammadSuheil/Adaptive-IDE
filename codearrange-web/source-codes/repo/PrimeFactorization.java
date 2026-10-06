// Description: Susunlah Output/Keluaran dari kode berikut, yang mencari semua faktor prima dari sebuah bilangan.
// Cognitive Load Rating: 7
// Bagus untuk output
// terlalu susah karena ada sqrt-nya

public class PrimeFactorization {
    public static void main(String[] args) {
        int n = 315;
        while (n % 2 == 0) {
            System.out.println(2);
            n /= 2;
        }
        for (int i = 3; i <= Math.sqrt(n); i += 2) {
            while (n % i == 0) {
                System.out.println(i);
                n /= i;
            }
        }
        if (n > 2) System.out.println(n);
    }
}

// Output
3
3
5
7
1
9