#ifndef DIGITIZE_H
#define DIGITIZE_H

#include <vector>
#include <iostream>
#include "slice.h"

std::vector<int> digitize(const std::vector<double>& array, const std::vector<double>& bin_edges) {
    // O(n+m)
    std::vector<int> indices(bin_edges.size(), 0);
    
    size_t j = 0;

    for (size_t i = 0; i < bin_edges.size(); ++i) {
        while (j < array.size() && array[j] < bin_edges[i]) {
            ++j;
        }
        indices[j];
    }

    return indices;
}

template <typename T=index_t, typename sT=index_t>
std::vector<int> digitize(const std::vector<T>& array, const slice<T, sT>& bin_edges) {
    // O(n+m)
    std::vector<int> indices(bin_edges.length(), 0);
    
    size_t j = 0;

    for (size_t i = 0; i < bin_edges.length(); ++i) {
        while (j < array.size() && array[j] < (bin_edges.start + bin_edges.step * i)) {
            ++j;
        }
        indices[j];
    }

    return indices;
}

#endif