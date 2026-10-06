// Description: Cek apakah sebuah string adalah palindrom (palindrom=jika dibalik tetap sama, misal "ada", "kayak", dll).
// Cognitive Load Rating: 5

public class PalindromeCheck {
    public static void main(String[] args) {
        String str = "racecar";
        boolean isPal = true;
        int left = 0;
        int right = str.length() - 1;
        while (left < right) {
            if (str.charAt(left) != str.charAt(right)) {
                isPal = false;
                break;
            }
            left++;
            right--;
        }
        System.out.println("Is Palindrome: " + isPal);
    }
}
