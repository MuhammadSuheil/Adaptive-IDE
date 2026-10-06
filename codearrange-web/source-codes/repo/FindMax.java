// Description: Cari dan tampilkan nilai Maksimum.
// Cognitive Load Rating: 3

public class FindMax {
    public static void main(String[] args) {
        int[] nums = {5, 12, 3, 9};
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > max) {
                max = nums[i];
            }
        }
        System.out.println("Max: " + max);
    }
}
