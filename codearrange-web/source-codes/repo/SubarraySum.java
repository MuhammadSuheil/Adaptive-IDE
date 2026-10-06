// Description: Finds a continuous subarray that adds up to a given target sum.
// Cognitive Load Rating: 8

public class SubarraySum {
    public static void main(String[] args) {
        int[] arr = {1, 4, 20, 3, 10, 5};
        int sum = 33, currentSum = arr[0], start = 0;
        for (int i = 1; i <= arr.length; i++) {
            while (currentSum > sum && start < i - 1) {
                currentSum = currentSum - arr[start];
                start++;
            }
            if (currentSum == sum) {
                System.out.println("Found between " + start + " and " + (i - 1));
                break;
            }
            if (i < arr.length) currentSum = currentSum + arr[i];
        }
    }
}
