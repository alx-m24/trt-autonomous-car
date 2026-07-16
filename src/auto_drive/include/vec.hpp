#pragma once
#include <cmath>
#include <cstdint>

template<typename Type>
struct vec2 {
    Type x;
    Type y;

    vec2() = default;
    vec2(Type n) : x(n), y(n) {}
    vec2(Type x, Type y) : x(x), y(y) {}

    float length() const {
        return std::sqrt((x * x) + (y * y));
    }

    template<typename Other_T>
    float distance(const vec2<Other_T>& other) const {
        return (*this - other).length();
    }

    template<typename Other_T>
    vec2<Type> operator+(const vec2<Other_T>& other) const {
        return { x + other.x, y + other.y };
    }
    template<typename Other_T>
    vec2<Type> operator-(const vec2<Other_T>& other) const {
        return { x - other.x, y - other.y };
    }
    template<typename Other_T>
    vec2<Type> operator*(const vec2<Other_T>& other) const {
        return { x * other.x, y * other.y };
    }
    template<typename Other_T>
    vec2<Type> operator*(const Other_T& other) const {
        return { x * other, y * other };
    }
    template<typename Other_T>
    vec2<Type> operator/(const vec2<Other_T>& other) const {
        return { x / other.x, y / other.y };
    }
    template<typename Other_T>
    vec2<Type> operator/(const Other_T& other) const {
        return { x / other, y / other };
    }
    bool operator==(const vec2<Type>& other) const {
        return other.x == x && other.y == y;
    }
};

using vec2f = vec2<float>;
using vec2i = vec2<int32_t>;
using vec2u = vec2<uint32_t>;

template<typename Type>
struct vec3 {
    Type x;
    Type y;
    Type z;

    vec3() = default;
    vec3(Type n) : x(n), y(n), z(n) {}
    vec3(Type x, Type y, Type z) : x(x), y(y), z(z) {}

    float length() const {
        return std::sqrt((x * x) + (y * y) + (z * z));
    }

    template<typename Other_T>
    float distance(const vec3<Other_T>& other) const {
        return (*this - other).length();
    }

    template<typename Other_T>
    vec3<Type> operator+(const vec3<Other_T>& other) const {
        return { x + other.x, y + other.y, z + other.z };
    }
    template<typename Other_T>
    vec3<Type> operator-(const vec3<Other_T>& other) const {
        return { x - other.x, y - other.y, z - other.z };
    }
    template<typename Other_T>
    vec3<Type> operator*(const vec3<Other_T>& other) const {
        return { x * other.x, y * other.y, z * other.z };
    }
    template<typename Other_T>
    vec3<Type> operator*(const Other_T& other) const {
        return { x * other, y * other, z * other };
    }
    template<typename Other_T>
    vec3<Type> operator/(const vec3<Other_T>& other) const {
        return { x / other.x, y / other.y, z / other.z };
    }
    template<typename Other_T>
    vec3<Type> operator/(const Other_T& other) const {
        return { x / other, y / other, z / other };
    }
    bool operator==(const vec3<Type>& other) const {
        return other.x == x && other.y == y && other.z == z;
    }
};

using vec3f = vec3<float>;
using vec3i = vec3<int32_t>;
using vec3u = vec3<uint32_t>;
