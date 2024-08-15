#include <string>
#include <iostream>
#include <Eigen/Dense>
#include "read_csv.h"

int main() {
    std::string filename = "C:\\Users\\evanw\\Sloth\\GOOG.csv";
    
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix = read_csv(filename);

    std::cout << "Eigen Matrix:" << std::endl;
    std::cout << matrix << std::endl;

    return 0;
}