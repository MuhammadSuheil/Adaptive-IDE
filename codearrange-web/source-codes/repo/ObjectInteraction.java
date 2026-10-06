// Description: Passes an object reference to another object's method.
// Cognitive Load Rating: 4

class Player {
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
}
