#include <algorithm>
#include <array>

namespace lar {

  template <class T>
  struct unordered_vector {
      std::vector<T> _data;
    public:
      using iterator = typename std::vector<T>::iterator;
      using const_iterator = typename std::vector<T>::const_iterator;

      T& operator[](std::size_t index) { return _data[index]; }
      const T& operator[](std::size_t index) const { return _data[index]; }


      void push_back(T value) { _data.push_back(value); }

      void pop_front() { _data[0] = _data.back(); pop_back(); }
      void pop_front(std::size_t k) {
        auto offset = _data.begin() + std::max(k, size() - k);
        std::copy(offset, _data.end(), _data.begin());
        pop_back(k);
      }
      
      void pop_back() { _data.pop_back(); }
      void pop_back(std::size_t k) { _data.resize(_data.size() - k); }

      void erase(std::size_t index) { _data[index] = _data.back(); pop_back(); }

      iterator clear() { _data.clear(); }

      std::size_t size() const { return _data.size(); }

      iterator begin() noexcept { return _data.begin(); }
      constexpr const_iterator begin() const noexcept { return _data.begin(); }

      iterator end() noexcept { return _data.end(); }
      constexpr const_iterator end() const noexcept { return _data.end(); }
  };
  
}