#pragma once

#include <memory>

#include "mat2f.h"

class KDTree {
private:
    class Node;
    typedef std::shared_ptr<Node> pNode;
    typedef std::pair<pNode, float> QueueElement;
    typedef std::vector<QueueElement> PriorityQueue;

public:
    KDTree(mat2f &data);

    void init();

    int find_nn(const range<float> &x, float *dist_sqr, float margin_sqr = -1) const;

private:
    pNode build_tree(int l, int r, int depth);
    void find_nn(const range<float> &x, const pNode &node, int *best, float *dist_sqr, int &visits_left,
            float margin_sqr, PriorityQueue &priority_queue, float cur_bin_dist_sqr) const;

    friend bool compare(const std::pair<pNode, float> &a, const std::pair<pNode, float> &b);
    void push_into_queue(PriorityQueue &queue, const QueueElement &elem) const;
    QueueElement pop_from_queue(PriorityQueue &queue) const;

public:
    int max_visits = 100;
    bool use_heap = true;

private:
    mat2f &data;
    int n;
    int d;
    pNode head;

    vec1i indices;

    class Node {
    public:
        Node(int index);
        Node(int index, int dim, float m);

        int index;

        int dim;
        float m;

        pNode left, right;
    };
};
