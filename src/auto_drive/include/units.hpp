#pragma once

template<typename Tag, typename T>
struct StrongType {
    T value;

    StrongType() = default;
    StrongType(T v) : value(v) {}

    T& operator=() {
        return value;
    }
    const T& operator=() const {
        return value;
    }
};

struct CmTag {};
struct MTag {};

using cm = StrongType<CmTag, int>;
using m  = StrongType<MTag, int>;

