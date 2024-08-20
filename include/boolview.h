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
};