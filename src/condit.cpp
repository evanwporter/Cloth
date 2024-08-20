#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <algorithm>

struct FilterResult {
    std::vector<bool> boolArray;
    int onesCount;

    FilterResult(const std::vector<bool>& boolArray, int onesCount)
        : boolArray(boolArray), onesCount(onesCount) {}
};

class MatrixFilter {
public:
    MatrixFilter(const Eigen::MatrixXd& data) : data(data) {}

    Eigen::MatrixXd filterData(const FilterResult& filterResult) {
        int numRows = filterResult.onesCount;
        int numCols = data.cols();

        Eigen::MatrixXd filteredData(numRows, numCols);
        int index = 0;

        for (int i = 0; i < data.rows(); ++i) {
            if (filterResult.boolArray[i]) {
                filteredData.row(index++) = data.row(i);
            }
        }

        return filteredData;
    }

private:
    Eigen::MatrixXd data;
};

FilterResult createFilterResult(const Eigen::MatrixXd& mat, int colIdx, double threshold) {
    std::vector<bool> boolArray(mat.rows(), false);
    int onesCount = 0;

    for (int i = 0; i < mat.rows(); ++i) {
        if (mat(i, colIdx) > threshold) {
            boolArray[i] = true;
            ++onesCount;
        }
    }

    return FilterResult(boolArray, onesCount);
}

int main() {
    Eigen::MatrixXd mat(5, 3);
    mat << 1, 2, 3,
           4, 5, 6,
           7, 8, 9,
           10, 11, 12,
           13, 14, 15;

    int colIdx = 0; // The column to filter on
    double threshold = 7.0;

    FilterResult filterResult = createFilterResult(mat, colIdx, threshold);

    MatrixFilter matrixFilter(mat);
    Eigen::MatrixXd filteredData = matrixFilter.filterData(filterResult);

    std::cout << "Filtered Matrix:\n" << filteredData << std::endl;

    return 0;
}
