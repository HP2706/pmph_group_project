#pragma once

#include <type_traits>
#include <cstdint>


// Generic UInt template for N-bit unsigned integers
template<unsigned int N>
struct UInt {
    static_assert(N > 0 && N <= 32, "Bit width must be between 1 and 32");
    static constexpr int bit_width = N;
    unsigned int value : N;  // N-bit unsigned value

    // Constructors
    UInt() : value(0) {}
    explicit UInt(unsigned int v) : value(v & ((1u << N) - 1)) {}  // Mask to N bits

    // Conversion operator to allow implicit conversion to uint32_t
    __host__ __device__ operator uint32_t() const { return static_cast<uint32_t>(value); }
};

// Base trait for bit width
template<typename T, typename = void>
struct has_bit_width : std::false_type {};

// Specialization for types with bit_width member
template<typename T>
struct has_bit_width<T, std::void_t<decltype(T::bit_width)>> : std::true_type {};

// Specializations for standard unsigned types
template<> struct has_bit_width<uint8_t> : std::true_type {};
template<> struct has_bit_width<uint16_t> : std::true_type {};
template<> struct has_bit_width<uint32_t> : std::true_type {};

// Specialization for UInt<N> template
template<unsigned int N>
struct has_bit_width<UInt<N>> : std::true_type {
    static constexpr int bit_width = N;
};

// Helper to get bit width
template<typename T, typename = void>
struct get_bit_width : std::integral_constant<int, 0> {};

template<typename T>
struct get_bit_width<T, std::enable_if_t<has_bit_width<T>::value>> {
    static constexpr int value = T::bit_width;
};

// Specializations for standard unsigned types
template<> struct get_bit_width<uint8_t> : std::integral_constant<int, 8> {};
template<> struct get_bit_width<uint16_t> : std::integral_constant<int, 16> {};
template<> struct get_bit_width<uint32_t> : std::integral_constant<int, 32> {};

// Type trait to check if T's bit width is <= 32
template<typename T, typename = void>
struct is_valid_bit_width : std::false_type {};

template<typename T>
struct is_valid_bit_width<T, std::enable_if_t<has_bit_width<T>::value>> :
    std::integral_constant<bool, (get_bit_width<T>::value <= 32)> {};

// Type trait to check if T can be safely zero-extended to uint32_t
template<typename T>
struct is_zero_extendable_to_uint32 {
    static constexpr bool value = 
        std::is_convertible<T, uint32_t>::value &&
        has_bit_width<T>::value &&
        is_valid_bit_width<T>::value;
};

// Specialization for UInt<N>
template<unsigned int N>
struct is_zero_extendable_to_uint32<UInt<N>> {
    static constexpr bool value = N <= 32;  // Explicitly check N is within bounds
};

// Add type alias to help with template deduction
template<typename T>
using remove_cv_ref = std::remove_cv_t<std::remove_reference_t<T>>;

// Helper type trait to detect if type is UInt<N>
template<typename T>
struct is_uint_n : std::false_type {};

template<unsigned int N>
struct is_uint_n<UInt<N>> : std::true_type {};

template<typename T>
inline constexpr bool is_uint_n_v = is_uint_n<remove_cv_ref<T>>::value;

// Alternative approach using a static member
template<typename T>
struct is_uint_n_v_helper {
    static constexpr bool value = is_uint_n<remove_cv_ref<T>>::value;
};
#define is_uint_n_v(T) (is_uint_n_v_helper<T>::value)
