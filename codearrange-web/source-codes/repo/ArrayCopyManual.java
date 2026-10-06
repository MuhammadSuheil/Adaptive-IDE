// Description: Manually copies elements from one array to another.
// Cognitive Load Rating: 5

public class ArrayCopyManual {
    public static void main(String[] args) {
        int[] source = {1, 2, 3, 4, 5};
        int[] dest = new int[source.length];
        for (int i = 0; i < source.length; i++) {
            dest[i] = source[i];
        }
    }
}
