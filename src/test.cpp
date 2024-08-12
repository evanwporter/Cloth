#include <iostream>

// Base class with utility functions
class UtilityBase {
protected:
    // Protected utility function
    void helperFunction1() {
        std::cout << "Helper Function 1 accessing DerivedClass1's attribute: " 
                  << derivedAttribute << std::endl;
    }

    void helperFunction2() {
        std::cout << "Helper Function 2" << std::endl;
    }

public:
    UtilityBase() = default;
    virtual ~UtilityBase() = default;
};

// Derived class inheriting from UtilityBase and NestedUtility
class DerivedClass1 : public UtilityBase {
protected:
    int derivedAttribute;  // Attribute defined in DerivedClass1

public:
    DerivedClass1(int val) : derivedAttribute(val) {}

    void useUtilityFunctions() {
        // Use functions from UtilityBase
        helperFunction1();
        helperFunction2();
    }
};

int main() {
    DerivedClass1 obj1(42);  // DerivedClass1 object with an attribute value of 42
    obj1.useUtilityFunctions();  // This will call helperFunction1 and helperFunction2

    return 0;
}
