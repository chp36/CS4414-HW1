#include "knn.hpp"
#include <vector>
#include <chrono>
#include <algorithm>

// Definition of static member
Embedding_T Node::queryEmbedding;


float distance(const Embedding_T &a, const Embedding_T &b)
{
    return std::abs(a - b);
}


constexpr float getCoordinate(Embedding_T e, size_t axis)
{
    return e;  // scalar case
}

// Build a balanced KD‐tree by splitting on median at each level.
Node* buildKD(std::vector<std::pair<Embedding_T,int>>& items, int depth) {
    /*
    TODO: Implement this function to build a balanced KD-tree.
    You should recursively construct the tree and return the root node.
    For now, this is a stub that returns nullptr.
    */

    if (items.empty()) return nullptr;

    int d = 1;
    int axis = depth % d;
    // items.sort(axis); // if equal, sort by axis + 1 % d (continuously until all the way through)
    
    std::sort(items.begin(), items.end(), [axis, d](const auto &a, const auto &b) {
        for (int i = 0; i < d; i++) {
            int dimension = (axis + i) % d;
            if (getCoordinate(a.first, dimension) < getCoordinate(b.first, dimension)) {
                return true;
            } else if (getCoordinate(a.first, dimension) > getCoordinate(b.first, dimension)) {
                return false;
            }
        }
        return false;
    });
    
    int medianIndex = items.size() / 2;
    auto median = items[medianIndex];

    Node *node = new Node(median.first, median.second);

    std::vector<std::pair<Embedding_T, int>> left(items.begin(), items.begin() + medianIndex);
    std::vector<std::pair<Embedding_T, int>> right(items.begin() + medianIndex + 1, items.end());
    
    node->left = buildKD(left, depth + 1);
    node->right = buildKD(right, depth + 1);

    return node;
}


void freeTree(Node *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}


void knnSearch(Node *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    /*
    TODO: Implement this function to perform k-nearest neighbors (k-NN) search on the KD-tree.
    You should recursively traverse the tree and maintain a max-heap of the K closest points found so far.
    For now, this is a stub that does nothing.
    */
    if (!node) return;
    int d = 1;
    int axis = depth % d;

    Node *near_node;
    Node *far_node;

    if (getCoordinate(Node::queryEmbedding, axis) < getCoordinate(node->embedding, axis)) {
        near_node = node->left;
        far_node = node->right;
    } else {
        near_node = node->right;
        far_node = node->left;
    }

    // Explore the nearest child
    knnSearch(near_node, depth + 1, K, heap);

    float dist = distance(node->embedding, Node::queryEmbedding);
    heap.push(std::make_pair(dist, node->idx));
    if (static_cast<int>(heap.size()) > K) {
        heap.pop();
    }

    float distance_coords = std::abs(getCoordinate(node->embedding, axis) - getCoordinate(Node::queryEmbedding, axis));
    float worst_dist = heap.top().first;

    // Explore the far child
    if (static_cast<int>(heap.size()) < K || distance_coords < worst_dist) {
        knnSearch(far_node, depth + 1, K, heap);
    }

    return;
}