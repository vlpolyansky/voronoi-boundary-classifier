#include <memory>

#include <algorithm>
#include <limits>

#include "KDTree.h"

KDTree::KDTree(mat2f &data) : data(data) {
    n = data.rows;
    d = data.cols;
}

void KDTree::init() {
    for (int i = 0; i < data.rows; i++) {
        indices.push_back(i);
    }
    head = build_tree(0, data.rows - 1, 0);
}

KDTree::pNode KDTree::build_tree(int l, int r, int depth) {
    if (l > r) {
        return nullptr;
    }
    if (l == r) {
        return std::make_shared<Node>(indices[l]);
    }
    int dim = depth % d;
    std::sort(indices.begin() + l, indices.begin() + r + 1, [&](int a, int b) {
        return data(a, dim) < data(b, dim);
    });
    int m = (l + r) / 2;
    pNode node(new Node(indices[m], dim, data(indices[m], dim)));
    node->left = build_tree(l, m, depth + 1);
    node->right = build_tree(m + 1, r, depth + 1);
    return node;
}

bool compare(const std::pair<KDTree::pNode, float> &a, const std::pair<KDTree::pNode, float> &b) {
    return a.second > b.second;
};

void KDTree::push_into_queue(KDTree::PriorityQueue &queue, const KDTree::QueueElement &elem) const {
    queue.push_back(elem);
    std::push_heap(queue.begin(), queue.end(), compare);
}

KDTree::QueueElement KDTree::pop_from_queue(KDTree::PriorityQueue &queue) const {
    std::pop_heap(queue.begin(), queue.end(), compare);
    KDTree::QueueElement elem = queue[queue.size() - 1];
    queue.pop_back();
    return elem;
}

inline float sqr(float a) {
    return a * a;
}

int KDTree::find_nn(const range<float> &x, float *dist_sqr, float margin_sqr) const {
    *dist_sqr = std::numeric_limits<float>::infinity();
    int best = -1;
    int visits_left = max_visits;
    PriorityQueue priority_queue;
    push_into_queue(priority_queue, std::make_pair(head, 0));
    while (!priority_queue.empty() && visits_left >= 0 && (margin_sqr < 0 || *dist_sqr >= margin_sqr)) {
        QueueElement elem = pop_from_queue(priority_queue);
        if (elem.second <= (margin_sqr < 0 ? *dist_sqr : margin_sqr)) {
            find_nn(x, elem.first, &best, dist_sqr, visits_left, margin_sqr, priority_queue, elem.second);
        }
    }
    return best;
}

void KDTree::find_nn(const range<float> &x, const KDTree::pNode &node, int *best, float *dist_sqr,
        int &visits_left, float margin_sqr, PriorityQueue &priority_queue, float cur_bin_dist_sqr) const {
    if (visits_left <= 0 || (margin_sqr >= 0 && *dist_sqr < margin_sqr)) {
        return;
    }
    if (node->dim == -1) {
        // leaf node
        visits_left--;
        float d = length_sqr(x - data[node->index]);
        if (d < *dist_sqr) {
            *dist_sqr = d;
            *best = node->index;
        }
        return;
    }
    pNode fst, snd;
    if (x[node->dim] <= node->m) {
        fst = node->left;
        snd = node->right;
    } else {
        fst = node->right;
        snd = node->left;
    }
    if (fst) {
        find_nn(x, fst, best, dist_sqr, visits_left, margin_sqr, priority_queue, cur_bin_dist_sqr);
    }
    float new_bin_dist_sqr = cur_bin_dist_sqr + sqr(x[node->dim] - node->m); // todo: this is only for a case [log n <= d]
    if (snd && new_bin_dist_sqr <= (margin_sqr < 0 ? *dist_sqr : margin_sqr)) {
        if (use_heap) {
            push_into_queue(priority_queue, std::make_pair(snd, new_bin_dist_sqr));
        } else {
            find_nn(x, snd, best, dist_sqr, visits_left, margin_sqr, priority_queue, new_bin_dist_sqr);
        }
    }

}

KDTree::Node::Node(int index) : index(index), dim(-1) {}

KDTree::Node::Node(int index, int dim, float m) : index(index), dim(dim), m(m) { }
