typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdRowMajor;

namespace py = pybind11;


template <typename T>
struct slice {
    T start, stop;
    int step;

    slice(T start_, T stop_, int step_) : start(start_), stop(stop_), step(step_) {
        if (step == 0) throw std::invalid_argument("Step cannot be zero");
    }

    void normalize(Eigen::Index length) {
        if (start < 0) start += length;
        if (stop < 0) stop += length;
        if (start < 0) start = 0;
        if (stop > length) stop = length;
    }

    Eigen::Index length() const {
        return (step > 0) ? (stop - start + step - 1) / step : (start - stop - step + 1) / (-step);
    }
    

    T get_start() const { return start; }
    T get_stop() const { return stop; }
    int get_step() const { return step; }

    std::string repr() const {
        std::ostringstream oss;
        oss << "slice(" << start << ", " << stop << ", " << step << ")";
        return oss.str();
    }
};

slice<int> combine_slices(const slice<int>& mask, const slice<int>& overlay, Eigen::Index length_mask) {
    int start = mask.start + (overlay.start * mask.step);
    int stop = mask.start + (overlay.stop * mask.step);
    int step = mask.step * overlay.step;
    return slice<int>(start, stop, step);
}

// Helper function to convert std::vector<T> to py::array
template <typename T>
py::array vector_to_numpy(const std::vector<T>& vec) {
    // Create a buffer info for the vector
    auto buf_info = py::buffer_info(
        (void*)vec.data(),                  // Pointer to buffer
        sizeof(T),                          // Size of one scalar
        py::format_descriptor<T>::format(), // Python struct-style format descriptor
        1,                                  // Number of dimensions
        { vec.size() },                     // Buffer dimensions
        { sizeof(T) }                       // Strides (in bytes) for each index
    );

    // Create a py::array using the buffer info
    return py::array(buf_info);
}

template <typename T>
MatrixXdRowMajor load_csv(const std::string &fname, const std::string &path, const bool quiet=true)
{
    // Modifjed from:
    // https://github.com/evanwporter/CAT/blob/main/DataHandler/dh.cpp
    
    std::vector<T> values;

    std::ifstream indata(path + fname + ".csv");
    if (!indata.is_open()) {
        throw std::runtime_error("Unable to open file: " + path + fname + ".csv");
    }

    std::string line;

    // HEADERS
    std::getline(indata, line);
    std::stringstream lineStream(line);
    std::vector<std::string> headers;
    std::string header;
    while (getline(lineStream, header, ',')) {
        headers.push_back(header);
    }

    std::size_t data_width = headers.size();
    unsigned int rows = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // DATA
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;

        for (unsigned int i = 0; i < data_width; ++i) {
            std::getline(lineStream, cell, ',');
            if (i > 0) { // Skip index for values
                values.push_back(std::stod(cell)); // TODO: Accept other types
            }
        }
        ++rows;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    if (!quiet) {
        std::cout << "Loaded " << fname << ". Time taken: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " microseconds." << std::endl;
    }

    return Map<MatrixXdRowMajor>(values.data(), rows, data_width - 1);
};