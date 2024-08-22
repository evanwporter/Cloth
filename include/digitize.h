#ifndef DIGITIZE_H
#define DIGITIZE_H

#include <vector>
#include <iostream>
#include <memory>

#include "../include/slice.h"

#ifndef INDEX_T
#define INDEX_T
#include <cstddef>
typedef std::ptrdiff_t index_t;
#endif

template <typename T = index_t, typename sT = index_t>
std::vector<index_t> digitize(const std::shared_ptr<std::vector<T>> array, const slice<T, sT>& bin_edges) {
    std::vector<index_t> indices;
    indices.reserve(bin_edges.length() + 1);

    index_t j = 0;
    index_t array_size = array->size();
    index_t last_index = -1;

    for (index_t i = 0; i < bin_edges.length(); ++i) {
        T current_bin_edge = bin_edges.start + bin_edges.step * i;

        while (j < array_size && array->at(j) < current_bin_edge) {
            ++j;
        }

        if (j != last_index) { 
            indices.push_back(j);
            last_index = j;
        }
    }

    if (j < array_size) {
        indices.push_back(array_size);
    }

    return indices;
}

#endif
