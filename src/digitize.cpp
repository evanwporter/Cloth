#include <vector>
#include <iostream>

std::vector<int> digitize(const std::vector<double>& array, const std::vector<double>& bin_edges) {
    std::vector<int> indices(array.size(), 0);
    size_t j = 0;

    for (size_t i = 0; i < array.size(); ++i) {
        while (j < bin_edges.size() && array[i] >= bin_edges[j]) {
            ++j;
        }
        indices[i] = j;
    }

    return indices;
}

int main() {
    std::vector<double> array = {0.5, 0.9, 0.91, 1.5, 2.5, 2.4, 3.1, 3.5, 4.5, 5.5};
    std::vector<double> bin_edges = {1.0, 2.0, 3.0, 4.0};

    std::vector<int> indices = digitize(array, bin_edges);

    for (int index : indices) {
        std::cout << index << " ";
    }

    return 0;
}
