
namespace lar {

namespace {

template <typename T>
class _NodeTraits {
  public:
    static constexpr std::size_t MAX_CHILDREN = 25;
};

// RegionTree<size_t> is used for testing purposes
// MAX_CHILDREN is set to 4 to make testing easier
template<>
class _NodeTraits<size_t> {
  public:
    static constexpr std::size_t MAX_CHILDREN = 4;
};

}

}