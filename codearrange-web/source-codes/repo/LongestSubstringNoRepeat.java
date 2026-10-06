// Description: Finds longest substring without repeating characters using a sliding window and index array.
// Cognitive Load Rating: 10

public class LongestSubstringNoRepeat {
    public static int longestUniqueSubsttr(String str) {
        int n = str.length();
        int res = 0;
        int[] lastIndex = new int[256];
        for (int i = 0; i < 256; i++) {
            lastIndex[i] = -1;
        }
        int i = 0;
        for (int j = 0; j < n; j++) {
            i = Math.max(i, lastIndex[str.charAt(j)] + 1);
            res = Math.max(res, j - i + 1);
            lastIndex[str.charAt(j)] = j;
        }
        return res;
    }
}
