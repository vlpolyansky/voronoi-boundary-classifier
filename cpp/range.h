#pragma once

#include <vector>

template<class T>
class range {
public:
    const T *begin, *end;

    range(T *begin, T *end);

//    range(std::vector<T> &vec);

    range(const std::vector<T> &vec);

    range(typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end);

    int size() const;

//    T& operator[] (int i);
    const T operator[] (int i) const;
};


template<class T>
range<T>::range(const std::vector<T> &vec) : begin(vec.begin().operator->()), end(vec.end().operator->()) {}

template<class T>
range<T>::range(T *begin, T *end) : begin(begin), end(end) {}

template<class T>
range<T>::range(const typename std::vector<T>::iterator begin, const typename std::vector<T>::iterator end) :
        begin(begin.operator->()), end(end.operator->()) {}

template<class T>
int range<T>::size() const {
    return static_cast<int>(end - begin);
}

//template<class T>
//T &range<T>::operator[](int i) {
//    return *(begin + i);
//}

template<class T>
const T range<T>::operator[](int i) const {
    return *(begin + i);
}

