
namespace lar {

namespace {

template <typename T>
class NodeTraits {
  public:
    static constexpr std::size_t MAX_CHILDREN = 25;
};

// RegionTree<int> is used for testing purposes
// MAX_CHILDREN is set to 4 to make testing easier
template<>
class NodeTraits<int> {
  public:
    static constexpr std::size_t MAX_CHILDREN = 4;
};

}

}