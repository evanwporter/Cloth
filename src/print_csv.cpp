#include <string>
#include <iostream>
#include <Eigen/Dense>
#include "read_csv.h"

int main() {
    std::string filename = "C:\\Users\\evanw\\Sloth\\GOOG.csv";
    
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix = readCSVToEigen(filename);

    std::cout << "Eigen Matrix:" << std::endl;
    std::cout << matrix << std::endl;

    return 0;
}

// #include <iostream>
// #include <Eigen/Dense>

// int main() {
//     Eigen::Matrix2d mat;
//     mat(0,0) = 3;
//     mat(1,0) = 2.5;
//     mat(0,1) = -1;
//     mat(1,1) = mat(1,0) + mat(0,1);
    
//     std::cout << "Matrix content:\n" << mat << std::endl;
//     return 0;
// }
