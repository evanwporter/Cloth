#include <iostream>
#include <memory>
#include <string>

using namespace std;

class Animal {
public:
    Animal() = default;

    virtual void move() {
        cout << "Animals can move" << endl;
    }

    virtual ~Animal() = default;
};

class Dog : public Animal {
public:
    string name;
    shared_ptr<int> color; // Use shared_ptr to share color between objects

    Dog(const string& name, int color) : name(name), color(make_shared<int>(color)) {}

    Dog(const Dog& other) : name(other.name), color(other.color) {
        // Copy constructor with shared color
    }

    void move() override {
        cout << "Dogs can walk and run" << endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " [dog] [name]" << endl;
        return 1;
    }

    string choice = argv[1];
    string name = argv[2];

    shared_ptr<Animal> animal;

    if (choice == "dog") {
        // Original Dog object
        auto originalDog = make_shared<Dog>(name, 2);

        // Shallow copy of the original Dog object
        Dog copiedDog = *originalDog;

        // Change the name of the copied Dog
        copiedDog.name = "NewDogName";

        // Change the color of the copied Dog (this will affect both original and copied dogs)
        *copiedDog.color = 3;

        // Display both Dog objects' colors
        cout << "Original Dog color: " << *originalDog->color << endl;
        cout << "Copied Dog color: " << *copiedDog.color << endl;

        // Display both Dog objects' names
        cout << "Original Dog name: " << originalDog->name << endl;
        cout << "Copied Dog name: " << copiedDog.name << endl;

        animal = make_shared<Dog>(copiedDog); // Assign copied Dog to the shared_ptr

    } else {
        cout << "Invalid choice. Please choose 'dog'." << endl;
        return 1;
    }

    animal->move(); // Call the move function on the selected animal

    return 0;
}



#include <iostream>
#include <memory>
#include <string>

using namespace std;

class Animal {
public:
    Animal() = default;

    virtual void move() {
        cout << "Animals can move" << endl;
    }

    // Pure virtual clone method with a default parameter
    virtual shared_ptr<Animal> clone(const string& newName = "") const = 0;

    virtual ~Animal() = default;
};

class Dog : public Animal {
public:
    shared_ptr<string> name;
    shared_ptr<int> color;

    // Constructor
    Dog(const string& name, int color)
        : name(make_shared<string>(name)), color(make_shared<int>(color)) {}

    // Copy Constructor
    Dog(const Dog& other, const string& newName = "")
        : name(make_shared<string>(newName.empty() ? *other.name : newName)),
          color(other.color) {}

    void move() override {
        cout << "Dogs can walk and run" << endl;
    }

    // Clone method using the copy constructor and passing a new name
    shared_ptr<Animal> clone(const string& newName = "") const override {
        return make_shared<Dog>(*this, newName);
    }
};

class Cat : public Animal {
public:
    shared_ptr<string> name;
    shared_ptr<string> color;

    // Constructor
    Cat(const string& name, const string& color)
        : name(make_shared<string>(name)), color(make_shared<string>(color)) {}

    // Copy Constructor
    Cat(const Cat& other, const string& newName = "")
        : name(make_shared<string>(newName.empty() ? *other.name : newName)),
          color(other.color) {}

    void move() override {
        cout << "Cats can walk and jump" << endl;
    }

    // Clone method using the copy constructor and passing a new name
    shared_ptr<Animal> clone(const string& newName = "") const override {
        return make_shared<Cat>(*this, newName);
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " [dog|cat] [name]" << endl;
        return 1;
    }

    string choice = argv[1];
    string name = argv[2];

    shared_ptr<Animal> animal;

    if (choice == "dog") {
        // Create a Dog object
        animal = make_shared<Dog>(name, 2);
    } else if (choice == "cat") {
        // Create a Cat object
        animal = make_shared<Cat>(name, "brown");
    } else {
        cout << "Invalid choice. Please choose either 'dog' or 'cat'." << endl;
        return 1;
    }

    // Clone the animal with a new name, regardless of whether it's a Dog or a Cat
    shared_ptr<Animal> clonedAnimal = animal->clone("NewAnimalName");

    // Print out the results to see the changes
    animal->move();
    clonedAnimal->move();

    auto dog = dynamic_pointer_cast<Dog>(clonedAnimal);
    if (dog) {
        cout << "Cloned Dog name: " << *dog->name << endl;
        cout << "Cloned Dog color: " << *dog->color << endl;
    }

    auto cat = dynamic_pointer_cast<Cat>(clonedAnimal);
    if (cat) {
        cout << "Cloned Cat name: " << *cat->name << endl;
        cout << "Cloned Cat color: " << *cat->color << endl;
    }

    return 0;
}
