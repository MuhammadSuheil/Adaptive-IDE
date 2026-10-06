// Description: Defines a class with fields and a simple method.
// Cognitive Load Rating: 3

class Car {
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
}
