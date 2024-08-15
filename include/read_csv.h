#ifndef READ_CSV_H
#define READ_CSV_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> read_csv(const std::string& filename) {
    std::vector<std::string> index;
    std::vector<std::string> headers;
    std::vector<double> data;
    int rows = 0;
    int cols = 0;    

    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open the file " << filename << std::endl;
        std::cout << "Could not open the file " << filename << std::endl;
        return Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();  // Return empty matrix
    }

    std::string line;
    
    // Process the first row (headers)
    if (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        bool isFirstCell = true;

        while (std::getline(lineStream, cell, ',')) {
            if (isFirstCell) {
                isFirstCell = false; // Skip the first column as it's considered the index header
            } else {
                headers.push_back(cell);
            }
        }
        cols = headers.size();
    }

    // Process the remaining rows (index + data)
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        bool isFirstCell = true;

        while (std::getline(lineStream, cell, ',')) {
            if (isFirstCell) {
                index.push_back(cell);  // First cell is the index
                isFirstCell = false;
            } else {
                data.push_back(std::stod(cell));  // Convert data to double
            }
        }
        rows++;
    }

    file.close();
    
    // Ensure that the data vector size matches the expected matrix size
    if (data.size() != rows * cols) {
        std::cerr << "Mismatch between data size and matrix dimensions" << std::endl;
        return Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();  // Return empty matrix
    }
    
    // Map the data vector to an Eigen matrix without copying
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(data.data(), rows, cols);
    
    return mat;
}

#endif