/*
Small file for demonstrating and testing deceig.h
*/
#include <iostream>
#include "lib/deceig.h"
#include <Eigen/Dense>

using Decimal = dec::decimal<2>;

int main() {
    Eigen::Matrix<Decimal, 2, 2> mat;
    mat(0, 0) = Decimal(1.1);
    mat(0, 1) = Decimal(2.2);
    mat(1, 0) = Decimal(3.3);
    mat(1, 1) = Decimal(4.4);

    Eigen::Matrix<Decimal, 2, 1> vec;
    vec(0) = Decimal("5.5");
    vec(1) = Decimal("6.6");

    Eigen::Matrix<Decimal, 2, 1> result = mat * vec;

    auto result2 = mat.sum();

    std::cout << "Matrix:\n" << mat << "\n";
    std::cout << "Vector:\n" << vec << "\n";
    std::cout << "Result:\n" << result << "\n";
    std::cout << "Result2:\n" << result2 << "\n";

    return 0;
}
