// Small file for demonstrating and testing deceig.h

#define CATCH_CONFIG_MAIN
#include "../lib/catch.hpp"
#include "../include/deceig.h"
#include <Eigen/Dense>

using Decimal = dec::decimal<2>;

TEST_CASE("Matrix and Vector operations using Decimal type with Eigen", "[Eigen][Decimal]") {
    Eigen::Matrix<Decimal, 2, 2> mat;
    mat(0, 0) = Decimal(1.1);
    mat(0, 1) = Decimal(2.2);
    mat(1, 0) = Decimal(3.3);
    mat(1, 1) = Decimal(4.4);

    Eigen::Matrix<Decimal, 2, 1> vec;
    vec(0) = Decimal("5.5");
    vec(1) = Decimal("6.6");

    Eigen::Matrix<Decimal, 2, 1> expected_result;
    expected_result(0) = Decimal("20.57");  // 1.1 * 5.5 + 2.2 * 6.6
    expected_result(1) = Decimal("47.19");  // 3.3 * 5.5 + 4.4 * 6.6

    Eigen::Matrix<Decimal, 2, 1> result = mat * vec;

    REQUIRE(result(0) == expected_result(0));
    REQUIRE(result(1) == expected_result(1));

    Decimal expected_sum = Decimal("11.00"); // 1.1 + 2.2 + 3.3 + 4.4

    Decimal result2 = mat.sum();

    REQUIRE(result2 == expected_sum);
}

