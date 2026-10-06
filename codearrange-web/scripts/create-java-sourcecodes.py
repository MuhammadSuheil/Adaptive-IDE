import os

java_files = [
    {
        "name": "HelloWorld",
        "rating": 1,
        "desc": "Prints a simple greeting to the console.",
        "code": """public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}"""
    },
    {
        "name": "Variables",
        "rating": 1,
        "desc": "Initializes and prints basic primitive variables.",
        "code": """public class Variables {
    public static void main(String[] args) {
        int age = 25;
        double price = 19.99;
        System.out.println(age + " " + price);
    }
}"""
    },
    {
        "name": "BasicArithmetic",
        "rating": 2,
        "desc": "Performs and prints basic arithmetic operations.",
        "code": """public class BasicArithmetic {
    public static void main(String[] args) {
        int a = 10, b = 3;
        int sum = a + b;
        int diff = a - b;
        System.out.println(sum + " " + diff);
    }
}"""
    },
    {
        "name": "SimpleIfElse",
        "rating": 2,
        "desc": "Checks a single condition using an if-else block.",
        "code": """public class SimpleIfElse {
    public static void main(String[] args) {
        int score = 85;
        if (score >= 50) {
            System.out.println("Pass");
        } else {
            System.out.println("Fail");
        }
    }
}"""
    },
    {
        "name": "EvenOdd",
        "rating": 2,
        "desc": "Uses the modulo operator to determine if a number is even or odd.",
        "code": """public class EvenOdd {
    public static void main(String[] args) {
        int number = 7;
        if (number % 2 == 0) {
            System.out.println("Even");
        } else {
            System.out.println("Odd");
        }
    }
}"""
    },
    {
        "name": "BasicForLoop",
        "rating": 3,
        "desc": "Iterates from 1 to 5 using a for loop.",
        "code": """public class BasicForLoop {
    public static void main(String[] args) {
        for (int i = 1; i <= 5; i++) {
            System.out.println("Count: " + i);
        }
    }
}"""
    },
    {
        "name": "BasicWhileLoop",
        "rating": 3,
        "desc": "Iterates using a while loop with a manual counter.",
        "code": """public class BasicWhileLoop {
    public static void main(String[] args) {
        int count = 5;
        while (count > 0) {
            System.out.println(count);
            count--;
        }
    }
}"""
    },
    {
        "name": "ArraySum",
        "rating": 3,
        "desc": "Calculates the sum of all elements in a static integer array.",
        "code": """public class ArraySum {
    public static void main(String[] args) {
        int[] nums = {2, 4, 6, 8};
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        System.out.println("Sum: " + sum);
    }
}"""
    },
    {
        "name": "FindMax",
        "rating": 3,
        "desc": "Finds the maximum value in an integer array.",
        "code": """public class FindMax {
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
}"""
    },
    {
        "name": "StringConcat",
        "rating": 2,
        "desc": "Concatenates multiple strings together.",
        "code": """public class StringConcat {
    public static void main(String[] args) {
        String first = "Java";
        String second = "Programming";
        String result = first + " " + second;
        System.out.println(result);
    }
}"""
    },
    {
        "name": "FactorialIterative",
        "rating": 4,
        "desc": "Calculates factorial using an iterative loop.",
        "code": """public class FactorialIterative {
    public static void main(String[] args) {
        int n = 5;
        long fact = 1;
        for (int i = 1; i <= n; i++) {
            fact *= i;
        }
        System.out.println("Factorial: " + fact);
    }
}"""
    },
    {
        "name": "FibonacciIterative",
        "rating": 4,
        "desc": "Generates the Fibonacci sequence iteratively.",
        "code": """public class FibonacciIterative {
    public static void main(String[] args) {
        int n = 10, a = 0, b = 1;
        System.out.print(a + " " + b);
        for (int i = 2; i < n; i++) {
            int next = a + b;
            System.out.print(" " + next);
            a = b;
            b = next;
        }
    }
}"""
    },
    {
        "name": "StringReverse",
        "rating": 4,
        "desc": "Reverses a string manually using a loop.",
        "code": """public class StringReverse {
    public static void main(String[] args) {
        String original = "hello";
        String reversed = "";
        for (int i = original.length() - 1; i >= 0; i--) {
            reversed += original.charAt(i);
        }
        System.out.println(reversed);
    }
}"""
    },
    {
        "name": "CountVowels",
        "rating": 4,
        "desc": "Iterates through a string to count vowels.",
        "code": """public class CountVowels {
    public static void main(String[] args) {
        String text = "education";
        int count = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c=='a' || c=='e' || c=='i' || c=='o' || c=='u') {
                count++;
            }
        }
        System.out.println("Vowels: " + count);
    }
}"""
    },
    {
        "name": "PalindromeCheck",
        "rating": 5,
        "desc": "Checks if a string is a palindrome using two pointers.",
        "code": """public class PalindromeCheck {
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
}"""
    },
    {
        "name": "LinearSearch",
        "rating": 4,
        "desc": "Searches for a target value in an array.",
        "code": """public class LinearSearch {
    public static void main(String[] args) {
        int[] arr = {10, 20, 30, 40};
        int target = 30;
        int index = -1;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                index = i;
                break;
            }
        }
        System.out.println("Found at: " + index);
    }
}"""
    },
    {
        "name": "Print2DArray",
        "rating": 4,
        "desc": "Uses nested loops to print a 2-dimensional array.",
        "code": """public class Print2DArray {
    public static void main(String[] args) {
        int[][] grid = { {1, 2}, {3, 4}, {5, 6} };
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                System.out.print(grid[i][j] + " ");
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "name": "MatrixAddition",
        "rating": 5,
        "desc": "Adds two 2D arrays (matrices) together.",
        "code": """public class MatrixAddition {
    public static void main(String[] args) {
        int[][] a = { {1, 2}, {3, 4} };
        int[][] b = { {5, 6}, {7, 8} };
        int[][] sum = new int[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                sum[i][j] = a[i][j] + b[i][j];
            }
        }
    }
}"""
    },
    {
        "name": "SimpleClass",
        "rating": 3,
        "desc": "Defines a class with fields and a simple method.",
        "code": """class Car {
    String model;
    int year;
    void start() {
        System.out.println(model + " started.");
    }
}
public class SimpleClass {
    public static void main(String[] args) {
        Car myCar = new Car();
        myCar.model = "Sedan";
        myCar.start();
    }
}"""
    },
    {
        "name": "ObjectInteraction",
        "rating": 4,
        "desc": "Passes an object reference to another object's method.",
        "code": """class Player {
    int health = 100;
    void takeDamage(int amount) {
        health -= amount;
    }
}
class Enemy {
    void attack(Player p) {
        p.takeDamage(10);
    }
}
public class ObjectInteraction {
    public static void main(String[] args) {
        Player p = new Player();
        Enemy e = new Enemy();
        e.attack(p);
    }
}"""
    },
    {
        "name": "BubbleSort",
        "rating": 6,
        "desc": "Sorts an array using the Bubble Sort algorithm.",
        "code": """public class BubbleSort {
    public static void main(String[] args) {
        int[] arr = {5, 2, 8, 1, 3};
        for (int i = 0; i < arr.length - 1; i++) {
            for (int j = 0; j < arr.length - 1 - i; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
}"""
    },
    {
        "name": "SelectionSort",
        "rating": 6,
        "desc": "Sorts an array using the Selection Sort algorithm.",
        "code": """public class SelectionSort {
    public static void main(String[] args) {
        int[] arr = {64, 25, 12, 22, 11};
        for (int i = 0; i < arr.length - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[j] < arr[minIdx]) {
                    minIdx = j;
                }
            }
            int temp = arr[minIdx];
            arr[minIdx] = arr[i];
            arr[i] = temp;
        }
    }
}"""
    },
    {
        "name": "InsertionSort",
        "rating": 6,
        "desc": "Sorts an array using the Insertion Sort algorithm.",
        "code": """public class InsertionSort {
    public static void main(String[] args) {
        int[] arr = {12, 11, 13, 5, 6};
        for (int i = 1; i < arr.length; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }
            arr[j + 1] = key;
        }
    }
}"""
    },
    {
        "name": "BinarySearch",
        "rating": 6,
        "desc": "Finds a target in a sorted array by halving the search space.",
        "code": """public class BinarySearch {
    public static void main(String[] args) {
        int[] arr = {2, 3, 4, 10, 40};
        int target = 10;
        int l = 0, r = arr.length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (arr[m] == target) {
                System.out.println("Found");
                break;
            }
            if (arr[m] < target) l = m + 1;
            else r = m - 1;
        }
    }
}"""
    },
    {
        "name": "PrimeCheck",
        "rating": 5,
        "desc": "Checks if a number is prime using a loop and modulo.",
        "code": """public class PrimeCheck {
    public static void main(String[] args) {
        int n = 29;
        boolean isPrime = true;
        if (n <= 1) isPrime = false;
        else {
            for (int i = 2; i <= Math.sqrt(n); i++) {
                if (n % i == 0) {
                    isPrime = false;
                    break;
                }
            }
        }
        System.out.println(isPrime);
    }
}"""
    },
    {
        "name": "GcdCalc",
        "rating": 5,
        "desc": "Calculates Greatest Common Divisor using Euclidean algorithm.",
        "code": """public class GcdCalc {
    public static void main(String[] args) {
        int a = 60, b = 48;
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        System.out.println("GCD: " + a);
    }
}"""
    },
    {
        "name": "LcmCalc",
        "rating": 6,
        "desc": "Calculates Least Common Multiple using a continuous loop.",
        "code": """public class LcmCalc {
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
}"""
    },
    {
        "name": "FactorialRecursive",
        "rating": 5,
        "desc": "Calculates factorial using method recursion.",
        "code": """public class FactorialRecursive {
    static int fact(int n) {
        if (n == 0) return 1;
        return n * fact(n - 1);
    }
    public static void main(String[] args) {
        System.out.println(fact(5));
    }
}"""
    },
    {
        "name": "FibonacciRecursive",
        "rating": 6,
        "desc": "Calculates Fibonacci number using method recursion.",
        "code": """public class FibonacciRecursive {
    static int fib(int n) {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }
    public static void main(String[] args) {
        System.out.println(fib(6));
    }
}"""
    },
    {
        "name": "ArrayCopyManual",
        "rating": 5,
        "desc": "Manually copies elements from one array to another.",
        "code": """public class ArrayCopyManual {
    public static void main(String[] args) {
        int[] source = {1, 2, 3, 4, 5};
        int[] dest = new int[source.length];
        for (int i = 0; i < source.length; i++) {
            dest[i] = source[i];
        }
    }
}"""
    },
    {
        "name": "MatrixTranspose",
        "rating": 7,
        "desc": "Swaps rows and columns of a 2D array.",
        "code": """public class MatrixTranspose {
    public static void main(String[] args) {
        int[][] original = { {1, 2, 3}, {4, 5, 6} };
        int rows = original.length;
        int cols = original[0].length;
        int[][] transposed = new int[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = original[i][j];
            }
        }
    }
}"""
    },
    {
        "name": "AnagramCheck",
        "rating": 7,
        "desc": "Checks if two strings are anagrams using character frequency arrays.",
        "code": """public class AnagramCheck {
    public static void main(String[] args) {
        String str1 = "listen";
        String str2 = "silent";
        if (str1.length() != str2.length()) return;
        int[] count = new int[256];
        for (int i = 0; i < str1.length(); i++) {
            count[str1.charAt(i)]++;
            count[str2.charAt(i)]--;
        }
        boolean isAnagram = true;
        for (int i = 0; i < 256; i++) {
            if (count[i] != 0) isAnagram = false;
        }
        System.out.println(isAnagram);
    }
}"""
    },
    {
        "name": "CaesarCipher",
        "rating": 7,
        "desc": "Encrypts a string by shifting character ASCII values.",
        "code": """public class CaesarCipher {
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
}"""
    },
    {
        "name": "PrimeFactorization",
        "rating": 7,
        "desc": "Finds and prints all prime factors of a given number.",
        "code": """public class PrimeFactorization {
    public static void main(String[] args) {
        int n = 315;
        while (n % 2 == 0) {
            System.out.print(2 + " ");
            n /= 2;
        }
        for (int i = 3; i <= Math.sqrt(n); i += 2) {
            while (n % i == 0) {
                System.out.print(i + " ");
                n /= i;
            }
        }
        if (n > 2) System.out.print(n);
    }
}"""
    },
    {
        "name": "RemoveDuplicatesArray",
        "rating": 7,
        "desc": "Removes duplicate elements in a sorted array by shifting.",
        "code": """public class RemoveDuplicatesArray {
    public static void main(String[] args) {
        int[] arr = {1, 2, 2, 3, 4, 4, 4, 5};
        int j = 0;
        for (int i = 0; i < arr.length - 1; i++) {
            if (arr[i] != arr[i + 1]) {
                arr[j++] = arr[i];
            }
        }
        arr[j++] = arr[arr.length - 1];
        for (int i = 0; i < j; i++) {
            System.out.print(arr[i] + " ");
        }
    }
}"""
    },
    {
        "name": "TicTacToeCheck",
        "rating": 8,
        "desc": "Evaluates a 2D array to check if there is a Tic-Tac-Toe winner.",
        "code": """public class TicTacToeCheck {
    public static void main(String[] args) {
        char[][] board = {
            {'X', 'O', 'X'},
            {'O', 'X', 'O'},
            {'O', 'O', 'X'}
        };
        char winner = '-';
        for (int i = 0; i < 3; i++) {
            if (board[i][0] == board[i][1] && board[i][1] == board[i][2]) winner = board[i][0];
            if (board[0][i] == board[1][i] && board[1][i] == board[2][i]) winner = board[0][i];
        }
        if (board[0][0] == board[1][1] && board[1][1] == board[2][2]) winner = board[0][0];
        if (board[0][2] == board[1][1] && board[1][1] == board[2][0]) winner = board[0][2];
        System.out.println("Winner: " + winner);
    }
}"""
    },
    {
        "name": "PascalTriangle",
        "rating": 8,
        "desc": "Generates Pascal's Triangle using a nested loop mathematical approach.",
        "code": """public class PascalTriangle {
    public static void main(String[] args) {
        int n = 5;
        for (int line = 1; line <= n; line++) {
            int C = 1;
            for (int i = 1; i <= line; i++) {
                System.out.print(C + " ");
                C = C * (line - i) / i;
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "name": "MergeSortedArrays",
        "rating": 8,
        "desc": "Merges two separately sorted arrays into a single sorted array.",
        "code": """public class MergeSortedArrays {
    public static void main(String[] args) {
        int[] arr1 = {1, 3, 5, 7};
        int[] arr2 = {2, 4, 6, 8};
        int[] merged = new int[arr1.length + arr2.length];
        int i = 0, j = 0, k = 0;
        while (i < arr1.length && j < arr2.length) {
            if (arr1[i] < arr2[j]) merged[k++] = arr1[i++];
            else merged[k++] = arr2[j++];
        }
        while (i < arr1.length) merged[k++] = arr1[i++];
        while (j < arr2.length) merged[k++] = arr2[j++];
    }
}"""
    },
    {
        "name": "SubarraySum",
        "rating": 8,
        "desc": "Finds a continuous subarray that adds up to a given target sum.",
        "code": """public class SubarraySum {
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
}"""
    },
    {
        "name": "FirstNonRepeatingChar",
        "rating": 8,
        "desc": "Finds the first non-repeating character in a string using nested loops.",
        "code": """public class FirstNonRepeatingChar {
    public static void main(String[] args) {
        String str = "swiss";
        for (int i = 0; i < str.length(); i++) {
            boolean unique = true;
            for (int j = 0; j < str.length(); j++) {
                if (i != j && str.charAt(i) == str.charAt(j)) {
                    unique = false;
                    break;
                }
            }
            if (unique) {
                System.out.println("First unique: " + str.charAt(i));
                break;
            }
        }
    }
}"""
    },
    {
        "name": "MatrixMultiplication",
        "rating": 8,
        "desc": "Multiplies two matrices requiring triple nested loops to compute dot products.",
        "code": """public class MatrixMultiplication {
    public static void main(String[] args) {
        int[][] a = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
        int[][] b = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
        int[][] c = new int[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                c[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
}"""
    },
    {
        "name": "SpiralMatrix",
        "rating": 9,
        "desc": "Traverses a 2D matrix in a spiral pattern tracking 4 boundary pointers.",
        "code": """public class SpiralMatrix {
    public static void main(String[] args) {
        int[][] mat = {{1,2,3},{4,5,6},{7,8,9}};
        int top = 0, bottom = 2, left = 0, right = 2;
        while (top <= bottom && left <= right) {
            for (int i = left; i <= right; i++) System.out.print(mat[top][i] + " ");
            top++;
            for (int i = top; i <= bottom; i++) System.out.print(mat[i][right] + " ");
            right--;
            if (top <= bottom) {
                for (int i = right; i >= left; i--) System.out.print(mat[bottom][i] + " ");
                bottom--;
            }
            if (left <= right) {
                for (int i = bottom; i >= top; i--) System.out.print(mat[i][left] + " ");
                left++;
            }
        }
    }
}"""
    },
    {
        "name": "SudokuValidator",
        "rating": 9,
        "desc": "Validates a 9x9 Sudoku grid using coordinate tracking and hash-based logic.",
        "code": """public class SudokuValidator {
    public static boolean isValid(int[][] board) {
        for (int i = 0; i < 9; i++) {
            int[] row = new int[10];
            int[] col = new int[10];
            int[] box = new int[10];
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != 0 && ++row[board[i][j]] > 1) return false;
                if (board[j][i] != 0 && ++col[board[j][i]] > 1) return false;
                int rowIdx = 3 * (i / 3) + j / 3;
                int colIdx = 3 * (i % 3) + j % 3;
                if (board[rowIdx][colIdx] != 0 && ++box[board[rowIdx][colIdx]] > 1) return false;
            }
        }
        return true;
    }
}"""
    },
    {
        "name": "GameOfLife",
        "rating": 9,
        "desc": "Calculates the next generation of Conway's Game of Life on a 2D grid.",
        "code": """public class GameOfLife {
    public void nextGen(int[][] board) {
        int m = board.length, n = board[0].length;
        int[][] next = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int lives = 0;
                for (int x = Math.max(0, i-1); x <= Math.min(m-1, i+1); x++) {
                    for (int y = Math.max(0, j-1); y <= Math.min(n-1, j+1); y++) {
                        lives += board[x][y];
                    }
                }
                lives -= board[i][j];
                if (board[i][j] == 1 && (lives == 2 || lives == 3)) next[i][j] = 1;
                else if (board[i][j] == 0 && lives == 3) next[i][j] = 1;
            }
        }
    }
}"""
    },
    {
        "name": "MazeSolverDFS",
        "rating": 9,
        "desc": "Solves a 2D grid maze using a recursive Depth-First Search approach.",
        "code": """public class MazeSolverDFS {
    public boolean solve(int[][] maze, int x, int y, int[][] sol) {
        int n = maze.length;
        if (x == n - 1 && y == n - 1 && maze[x][y] == 1) {
            sol[x][y] = 1;
            return true;
        }
        if (x >= 0 && x < n && y >= 0 && y < n && maze[x][y] == 1) {
            sol[x][y] = 1;
            if (solve(maze, x + 1, y, sol)) return true;
            if (solve(maze, x, y + 1, sol)) return true;
            sol[x][y] = 0;
        }
        return false;
    }
}"""
    },
    {
        "name": "StringPermutations",
        "rating": 9,
        "desc": "Generates all permutations of a string recursively with backtracking.",
        "code": """public class StringPermutations {
    public static void permute(String str, int l, int r) {
        if (l == r) System.out.println(str);
        else {
            for (int i = l; i <= r; i++) {
                str = swap(str, l, i);
                permute(str, l + 1, r);
                str = swap(str, l, i);
            }
        }
    }
    public static String swap(String a, int i, int j) {
        char[] charArray = a.toCharArray();
        char temp = charArray[i];
        charArray[i] = charArray[j];
        charArray[j] = temp;
        return String.valueOf(charArray);
    }
}"""
    },
    {
        "name": "NQueensBacktracking",
        "rating": 10,
        "desc": "Solves the N-Queens problem on a chess board using recursive backtracking.",
        "code": """public class NQueensBacktracking {
    boolean isSafe(int board[][], int row, int col) {
        for (int i = 0; i < col; i++) if (board[row][i] == 1) return false;
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) if (board[i][j] == 1) return false;
        for (int i = row, j = col; j >= 0 && i < board.length; i++, j--) if (board[i][j] == 1) return false;
        return true;
    }
    boolean solveNQUtil(int board[][], int col) {
        if (col >= board.length) return true;
        for (int i = 0; i < board.length; i++) {
            if (isSafe(board, i, col)) {
                board[i][col] = 1;
                if (solveNQUtil(board, col + 1)) return true;
                board[i][col] = 0;
            }
        }
        return false;
    }
}"""
    },
    {
        "name": "MergeSortAlgorithm",
        "rating": 10,
        "desc": "Implements the Divide and Conquer Merge Sort algorithm manually tracking sub-array states.",
        "code": """public class MergeSortAlgorithm {
    void merge(int arr[], int l, int m, int r) {
        int n1 = m - l + 1, n2 = r - m;
        int L[] = new int[n1], R[] = new int[n2];
        for (int i = 0; i < n1; ++i) L[i] = arr[l + i];
        for (int j = 0; j < n2; ++j) R[j] = arr[m + 1 + j];
        int i = 0, j = 0, k = l;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) arr[k++] = L[i++];
            else arr[k++] = R[j++];
        }
        while (i < n1) arr[k++] = L[i++];
        while (j < n2) arr[k++] = R[j++];
    }
    void sort(int arr[], int l, int r) {
        if (l < r) {
            int m = l + (r - l) / 2;
            sort(arr, l, m);
            sort(arr, m + 1, r);
            merge(arr, l, m, r);
        }
    }
}"""
    },
    {
        "name": "QuickSortAlgorithm",
        "rating": 10,
        "desc": "Implements Quick Sort algorithm requiring pivoting and recursive partition management.",
        "code": """public class QuickSortAlgorithm {
    int partition(int arr[], int low, int high) {
        int pivot = arr[high];
        int i = (low - 1);
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        return i + 1;
    }
    void sort(int arr[], int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            sort(arr, low, pi - 1);
            sort(arr, pi + 1, high);
        }
    }
}"""
    },
    {
        "name": "LongestSubstringNoRepeat",
        "rating": 10,
        "desc": "Finds longest substring without repeating characters using a sliding window and index array.",
        "code": """public class LongestSubstringNoRepeat {
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
}"""
    }
]

def main():
    print("Generating 50 Java source files...")
    for data in java_files:
        filename = f"{data['name']}.java"
        content = f"// Description: {data['desc']}\n// Cognitive Load Rating: {data['rating']}\n\n{data['code']}\n"
        
        with open(filename, 'w') as f:
            f.write(content)
            
    print(f"Success! Generated {len(java_files)} files in the current directory.")

if __name__ == "__main__":
    main()