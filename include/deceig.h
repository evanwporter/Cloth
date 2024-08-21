// deceig.h

/*
Defines decimal a custom scalar type for Eigen.

References:
https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
https://stackoverflow.com/questions/59747377/custom-scalar-type-in-eigen
*/

#ifndef DECIMAL_EIGEN_H
#define DECIMAL_EIGEN_H

#include "../lib/decimal.h"
#include <Eigen/Core>

namespace Eigen {
    template<int Precision>
    struct NumTraits<dec::decimal<Precision>> : GenericNumTraits<dec::decimal<Precision>> {
        typedef dec::decimal<Precision> Real;
        typedef dec::decimal<Precision> NonInteger;
        typedef dec::decimal<Precision> Nested;

        static inline Real epsilon() {
            // Return a small value that represents the precision limit of the decimal type
            return dec::decimal<Precision>(0.0001); // Adjust this value according to your needs
        }

        static inline Real dummy_precision() {
            // Provide a dummy precision, typically a small value
            return dec::decimal<Precision>(0.0001); // Adjust this value according to your needs
        }

        static inline int digits10() {
            // Returns the number of base-10 digits that can be represented without loss of precision
            return Precision;
        }

        enum {
            IsInteger = 0,
            IsSigned = 1,
            IsComplex = 0,
            RequireInitialization = 1,
            ReadCost = 1,
            AddCost = 1,
            MulCost = 1
        };
    };

    namespace internal {
        // Scalar addition
        template<int Precision>
        struct scalar_sum_op<dec::decimal<Precision>, dec::decimal<Precision>> {
            EIGEN_EMPTY_STRUCT_CTOR(scalar_sum_op)
            typedef dec::decimal<Precision> result_type;
            EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
            result_type operator()(const dec::decimal<Precision>& a, const dec::decimal<Precision>& b) const {
                return a + b;
            }
        };

        // Scalar multiplication
        template<int Precision>
        struct scalar_product_op<dec::decimal<Precision>, dec::decimal<Precision>> {
            EIGEN_EMPTY_STRUCT_CTOR(scalar_product_op)
            typedef dec::decimal<Precision> result_type;
            EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
            result_type operator()(const dec::decimal<Precision>& a, const dec::decimal<Precision>& b) const {
                return a * b;
            }
        };

        // Scalar quotient
        template<int Precision>
        struct scalar_quotient_op<dec::decimal<Precision>, dec::decimal<Precision>> {
            EIGEN_EMPTY_STRUCT_CTOR(scalar_quotient_op)
            typedef dec::decimal<Precision> result_type;
            EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
            result_type operator()(const dec::decimal<Precision>& a, const dec::decimal<Precision>& b) const {
                return a / b;
            }
        };

        // Scalar min
        template<int Precision>
        struct scalar_min_op<dec::decimal<Precision>, dec::decimal<Precision>> {
            EIGEN_EMPTY_STRUCT_CTOR(scalar_min_op)
            typedef dec::decimal<Precision> result_type;
            EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
            result_type operator()(const dec::decimal<Precision>& a, const dec::decimal<Precision>& b) const {
                return std::min(a, b);
            }
        };

        // Scalar max
        template<int Precision>
        struct scalar_max_op<dec::decimal<Precision>, dec::decimal<Precision>> {
            EIGEN_EMPTY_STRUCT_CTOR(scalar_max_op)
            typedef dec::decimal<Precision> result_type;
            EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
            result_type operator()(const dec::decimal<Precision>& a, const dec::decimal<Precision>& b) const {
                return std::max(a, b);
            }
        };
    }
}

#endif
