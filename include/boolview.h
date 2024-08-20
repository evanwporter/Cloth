#ifndef BOOLVIEW_T
#define BOOLVIEW_T

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <algorithm>

class BoolView {
public:
    
    std::vector<bool> mask_;
    int ones_count_;

    BoolView(std::vector<bool> mask, int ones_count)
        : mask_(std::move(mask)), ones_count_(ones_count) {}

    Eigen::MatrixXd apply(const Eigen::MatrixXd& data) const {
        Eigen::MatrixXd filtered_data(ones_count_, data.cols());
        int index = 0;
        for (int i = 0; i < data.rows(); ++i) {
            if (mask_[i]) {
                filtered_data.row(index++) = data.row(i);
            }
        }
        return filtered_data;
    }

    Eigen::VectorXd apply(const Eigen::VectorXd& data) const {
        Eigen::VectorXd filtered_data(ones_count_);
        int index = 0;
        for (int i = 0; i < data.size(); ++i) {
            if (mask_[i]) {
                filtered_data[index++] = data(i);
            }
        }
        return filtered_data;
    }
};

#endif