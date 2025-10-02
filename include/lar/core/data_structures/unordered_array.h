#include <algorithm>
#include <utility>

namespace lar {

  template <class T, std::size_t N>
  struct unordered_array {
      T _data[N];
      std::size_t _size;
    public:
      unordered_array() : _size(0) {}

      using iterator = T*;
      using const_iterator = const T*;

      T& operator[](std::size_t index) { return _data[index]; }
      const T& operator[](std::size_t index) const { return _data[index]; }

      void push_back(T&& value) { _data[_size++] = std::move(value); }
      void push_back(const T& value) { _data[_size++] = value; }

      void pop_front() { _data[0] = std::move(_data[--_size]); }
      void pop_front(std::size_t k) {
        auto end = _data + _size;
        _size -= k;
        std::size_t offset = std::max(k, _size);
        std::move(_data + offset, end, _data);
      }

      void pop_back() { _size--; }
      void pop_back(std::size_t k) { _size -= k; }

      void erase(std::size_t index) { _data[index] = std::move(_data[--_size]); }

      void clear() { _size = 0; }

      std::size_t size() const { return _size; }

      iterator begin() noexcept { return _data; }
      constexpr const_iterator begin() const noexcept { return _data; }

      iterator end() noexcept { return _data + _size; }
      constexpr const_iterator end() const noexcept { return _data + _size; }
  };
  
}