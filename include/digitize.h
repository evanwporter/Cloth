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
    indices.reserve(bin_edges.length());

    size_t j = 0;
    size_t array_size = array->size();

    for (size_t i = 0; i < bin_edges.length(); ++i) {
        T current_bin_edge = bin_edges.start + bin_edges.step * i;

        while (j < array_size && array->at(j) < current_bin_edge) {
            ++j;
        }
        indices.push_back(static_cast<index_t>(j));
    }

    return indices;
}

#endif
