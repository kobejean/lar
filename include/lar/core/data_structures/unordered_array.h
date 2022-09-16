#include <algorithm>
#include <array>

namespace lar {

  template <class T, std::size_t N>
  struct unordered_array {
      std::array<T, N> _data;
      std::size_t _size{0};
    public:
      using iterator = typename std::array<T, N>::iterator;
      using const_iterator = typename std::array<T, N>::const_iterator;

      T& operator[](std::size_t index) { return _data[index]; }
      const T& operator[](std::size_t index) const { return _data[index]; }


      void push_back(T value) { _data[_size++] = value; }

      void pop_back() { _size--; }
      void pop_back(std::size_t k) { _size -= k; }

      void erase(std::size_t index) { _data[index] = _data[--_size]; }

      iterator clear() { _size = 0; }


      std::size_t size() const { return _size; }

      iterator begin() noexcept { return _data.begin(); }
      constexpr const_iterator begin() const noexcept { return _data.begin(); }

      iterator end() noexcept { return _data.begin() + _size; }
      constexpr const_iterator end() const noexcept { return _data.begin() + _size; }
  };
  
}