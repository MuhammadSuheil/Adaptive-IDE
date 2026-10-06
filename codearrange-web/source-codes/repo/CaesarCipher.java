// Description: Encrypts a string by shifting character ASCII values.
// Cognitive Load Rating: 7

public class CaesarCipher {
    public static void main(String[] args) {
        String text = "hello";
        int shift = 3;
        String result = "";
        for (int i = 0; i < text.length(); i++) {
            char ch = (char)(((int)text.charAt(i) + shift - 97) % 26 + 97);
            result += ch;
        }
        System.out.println(result);
    }
}
