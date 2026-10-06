// Description: Calculates the sum of all elements in a static integer array.
// Cognitive Load Rating: 3

public class ArraySum {
    public static void main(String[] args) {
        int[] nums = {2, 4, 6, 8};
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        System.out.println("Sum: " + sum);
    }
}
