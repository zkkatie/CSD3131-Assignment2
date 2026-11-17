// Graph Cut seam finder (min-cut / max-flow)
//
// GOAL: find the cheapest vertical seam by turning it into a min-cut problem.
//
// THE GRAPH (H x W energy map):
//   - one node per pixel, plus a SOURCE and a SINK
//   - SOURCE -> every pixel in column 0               capacity INF
//   - pixel (i,j) -> right neighbour (i,j+1)          capacity e[i][j]
//         cutting this edge = removing pixel (i,j)
//   - pixel (i,j+1) -> left neighbour (i,j)           capacity INF
//         makes each row get cut exactly ONCE
//   - pixel (i,j) -> (i-1,j-1) and (i,j) -> (i+1,j-1) capacity INF
//         makes the seam CONNECTED (rows differ by at most 1 column)
//   - every pixel in the last column -> SINK          capacity e[i][W-1]
//         (the PDF says INF here, but then the last column could never be
//          removed, so we use its energy instead)
//
// WHY IT WORKS: every finite cut cuts exactly one "right" edge per row, and
// those cut pixels form a connected seam. The cut's capacity = sum of their
// energies = the seam's cost. Max-flow min-cut theorem: max flow = min cut,
// so the min cut is the cheapest seam.
//
// THE ALGORITHM:
//   1. Build the graph.
//   2. Run max-flow (Edmonds-Karp: keep pushing flow along the shortest path
//      from SOURCE to SINK, found by BFS, until no path is left).
//   3. BFS from SOURCE using edges that still have capacity left. The pixels
//      reached are the source side of the min cut. In each row, the rightmost
//      reached pixel is the seam pixel.
//
// COST per seam, V = H*W nodes, E = number of edges (about 6*H*W):
//   building the graph  O(V + E)
//   Edmonds-Karp        O(V * E^2) worst case
//   DP, for comparison  O(H * W)
#include <algorithm>
#include <queue>
#include <vector>

#include "seamcarve/methods.hpp"

namespace {

const double INF = 1e18;   // "infinite" capacity
const double EPS = 1e-9;   // capacity <= EPS counts as 0 (doubles are not exact)

// One directed edge. Every edge has a partner edge going the other way
// (its "reverse"), used to undo flow. Both live in the same list `edges`.
struct Edge {
    int to;        // node this edge points to
    double cap;    // capacity LEFT (residual capacity)
    int reverse;   // index of the partner edge in `edges`
};

struct Graph {
    std::vector<Edge> edges;               // all edges
    std::vector<std::vector<int>> out;     // out[node] = indices of edges leaving node
};

// Add edge u -> v with capacity c, plus its partner v -> u with capacity back_cap.
void add_edge(Graph& g, int u, int v, double c, double back_cap) {
    int a = (int)g.edges.size();      // index of u -> v
    int b = a + 1;                    // index of v -> u
    g.edges.push_back({v, c, b});
    g.edges.push_back({u, back_cap, a});
    g.out[u].push_back(a);
    g.out[v].push_back(b);
}

// BFS from `start` using only edges with capacity left.
// parent_edge[v] = index of the edge used to reach v (-1 = not reached).
// Returns true if `goal` was reached.
bool bfs(const Graph& g, int start, int goal, std::vector<int>& parent_edge) {
    int n = (int)g.out.size();
    parent_edge.assign(n, -1);
    std::vector<bool> visited(n, false);
    std::queue<int> q;
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int k : g.out[u]) {
            const Edge& e = g.edges[k];
            if (e.cap > EPS && !visited[e.to]) {
                visited[e.to] = true;
                parent_edge[e.to] = k;
                q.push(e.to);
            }
        }
    }
    return visited[goal];
}

// Edmonds-Karp max-flow. Changes the capacities in `g` into residual capacities.
void max_flow(Graph& g, int source, int sink) {
    std::vector<int> parent_edge;

    // Repeat while there is still a path from source to sink.
    while (bfs(g, source, sink, parent_edge)) {
        // 1. Find the bottleneck: smallest capacity along the path.
        //    Walk backwards from sink to source using parent_edge.
        double push = INF;
        int v = sink;
        while (v != source) {
            int k = parent_edge[v];
            push = std::min(push, g.edges[k].cap);
            v = g.edges[g.edges[k].reverse].to;   // the node this edge came from
        }

        // 2. Push that much flow along the path.
        v = sink;
        while (v != source) {
            int k = parent_edge[v];
            g.edges[k].cap -= push;                     // less capacity forward
            g.edges[g.edges[k].reverse].cap += push;    // more capacity backward (can undo)
            v = g.edges[g.edges[k].reverse].to;
        }
    }
}

std::vector<int> find_seam(const cv::Mat& energy) {
    int h = energy.rows;
    int w = energy.cols;
    std::vector<int> seam(h, 0);
    if (h == 0 || w <= 1) return seam;   // only one column: must remove column 0

    // Node numbers: pixel (i, j) is node i*w + j, then source and sink.
    int source = h * w;
    int sink = h * w + 1;

    // ---- Step 1: build the graph ----
    Graph g;
    g.out.resize(h * w + 2);

    for (int i = 0; i < h; ++i) {
        // source -> first pixel of the row
        add_edge(g, source, i * w + 0, INF, 0);

        // pixel -> right neighbour (cost e[i][j]).
        // Its partner edge (right -> left) gets INF: that IS the backward edge.
        for (int j = 0; j + 1 < w; ++j)
            add_edge(g, i * w + j, i * w + (j + 1), energy.at<double>(i, j), INF);

        // last pixel of the row -> sink (cost e[i][w-1])
        add_edge(g, i * w + (w - 1), sink, energy.at<double>(i, w - 1), 0);

        // diagonal edges going left, to the row above and the row below
        for (int j = 1; j < w; ++j) {
            if (i > 0)     add_edge(g, i * w + j, (i - 1) * w + (j - 1), INF, 0);
            if (i + 1 < h) add_edge(g, i * w + j, (i + 1) * w + (j - 1), INF, 0);
        }
    }

    // ---- Step 2: max flow ----
    max_flow(g, source, sink);

    // ---- Step 3: find the cut ----
    // After max-flow the sink cannot be reached. BFS once more: the nodes
    // reached form the source side of the min cut.
    std::vector<int> parent_edge;
    bfs(g, source, sink, parent_edge);

    for (int i = 0; i < h; ++i) {
        // Column 0 is always reached (source -> column 0 is INF).
        // Walk right while the next pixel is also reached.
        int j = 0;
        while (j + 1 < w && parent_edge[i * w + (j + 1)] != -1) ++j;
        seam[i] = j;   // rightmost reached pixel = pixel to remove
    }
    return seam;
}

}  // namespace

REGISTER_SEAM_FINDER("graphcut", find_seam);