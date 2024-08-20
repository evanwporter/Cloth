#ifndef BOOLVIEW_T
#define BOOLVIEW_T

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <algorithm>

#ifndef MATRIX_RM_T
#define MATRIX_RM_T
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
#endif

class BoolView {
public:
    
    std::vector<bool> mask_;
    int ones_count_;

    BoolView(std::vector<bool> mask, int ones_count)
        : mask_(std::move(mask)), ones_count_(ones_count) {}

    MatrixXdRowMajor apply(const MatrixXdRowMajor& data) const {
        MatrixXdRowMajor filtered_data(ones_count_, data.cols());
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

    std::string to_string() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < mask_.size(); ++i) {
            oss << (mask_[i] ? "True" : "False");
            if (i < mask_.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]";
        return oss.str();
    }
};

#endif