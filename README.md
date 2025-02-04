---
body_class: markdown-body
highlight_style: default
pdf_options:
  format: A4
  margin: 5mm
css: |-
  .markdown-body { font-size: 11px; }
---

<!-- Booklet print: https://bookbinder.app (A4, Duplex, 2 PPS, Booklet) -->

# Notebook

As complexidades são estimadas e às vezes eu não incluo todas as variáveis!

# Sumário

* Template
* Flags
* Debug
* Algoritmos
  * Árvores
    * Binary lifting
    * Centróide
    * Euler tour
    * Menor ancestral comum (LCA)
  * Geometria
    * Ângulo entre segmentos
    * Distância entre pontos
    * Envoltório convexo
    * Mediatriz
    * Orientação de ponto
    * Rotação de ponto
  * Grafos
    * Bellman-Ford
    * BFS 0/1
    * Caminho euleriano
    * Dijkstra
    * Floyd-Warshall
    * Kosaraju
    * Kruskal (Árvore geradora mínima)
    * Ordenação topológica
    * Max flow/min cut (Dinic)
  * Outros
    * Maior subsequência comum (LCS)
    * Maior subsequência crescente (LIS)
  * Matemática
    * Coeficiente binomial 
    * Conversão de base
    * Crivo de Eratóstenes
    * Divisores
    * Fatoração
    * Quantidade de divisores
    * Permutação com repetição
  * Strings
    * Distância de edição
    * Z-Function
    * Ocorrências de substring
* Estruturas
  * Árvores
    * BIT tree 2D
    * Disjoint set union
    * Red-Black tree
    * Segment tree
    * Wavelet tree
  * Geometria
    * Círculo
    * Reta
    * Segmento
    * Triângulo
    * Polígono
  * Matemática
    * Matriz
  * Strings
    * Hash
    * Suffix Automaton
  * Outros
    * Soma de prefixo 2D
    * Soma de prefixo 3D
* Utils
  * Aritmética modular
  * Big integer
  * Ceil division
  * Conversão de índices
  * Compressão de coordenadas
  * Fatos
  * Igualdade flutuante
  * Intervalos com soma S
  * Kadane
  * Overflow check
  * Pares com gcd x
  * Próximo maior/menor elemento
  * Soma de todos os intervalos

### Template

```c++
#include <bits/stdc++.h>
using namespace std;

#ifdef croquete  // BEGIN TEMPLATE ----------------------|
#include "dbg/dbg.h"
#else
#define dbg(...)
#endif
#define ll           long long
#define vll          vector<ll>
#define vvll         vector<vll>
#define pll          pair<ll, ll>
#define vpll         vector<pll>
#define all(xs)      xs.begin(), xs.end()
#define found(x, xs) (xs.find(x) != xs.end())
#define rep(i, a, b) for (ll i = (a); i < (ll)(b); ++i)
#define per(i, a, b) for (ll i = (a); i >= (ll)(b); --i)
#define eb           emplace_back
#define cinj         (cin.iword(0)  = 1, cin)
#define coutj        (cout.iword(0) = 1, cout)
template <typename T>  // read vector
istream& operator>>(istream& in, vector<T>& xs) {
    assert(!xs.empty());
    rep(i, in.iword(0), xs.size()) in >> xs[i];
    in.iword(0) = 0;
    return in;
} template <typename T>  // print vector
ostream& operator<<(ostream& os, vector<T>& xs) {
    rep(i, os.iword(0), xs.size() - 1) os << xs[i] << ' ';
    os.iword(0) = 0;
    return os << xs.back();
} void solve();
signed main() {
    cin.tie(0)->sync_with_stdio(0);
    ll t = 1;
    // cin >> t;
    while (t--) solve();
}  // END TEMPLATE --------------------------------------|

void solve() {
}
```

### Outros defines

```c++
// BEGIN EXTRAS ----------------------------------------|
#define vvpll vector<vpll>
#define tll   tuple<ll, ll, ll>
#define vtll  vector<tll>
#define pd    pair<double, double>
#define vb    vector<bool>
#define x     first
#define y     second
map<char, pll> ds1 {
    {'R', {0, 1}},  {'D', {1, 0}},
    {'L', {0, -1}}, {'U', {-1, 0}}
};
vpll ds2 { {0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };
vpll ds3 { {1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1} };
// END EXTRAS ------------------------------------------|
```

# Flags

`g++ -g -O2 -std=c++20 -fsanitize=undefined -fno-sanitize-recover -Wall -Wextra
 -Wshadow -Wno-sign-compare -Wconversion -Wno-sign-conversion -Wduplicated-cond
 -Winvalid-pch -Dcroquete -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -D_FORTIFY_SOURCE=2`

# Debug

```c++
#pragma once
#include <bits/stdc++.h>
using namespace std;
namespace DBG {
    template <typename T>
    void C(T x, int n = 4)  { cerr << fixed << "\e[9" << n << 'm' << x << "\e[m"; }
    void p(char x)          { C("'" + string({x}) + "'", 3); }
    void p(string x)        { C('"' + x + '"', 3); }
    void p(bool x)          { x ? C('T', 2) : C('F', 1); }
    void p(vector<bool> xs) { for (bool x : xs) p(x); }
    static int m = 0;
    template <typename T>
    void p(T x) {
        int i = 0;
        if constexpr (requires { begin(x); }) {  // nested iterable
            C('{');
            if (size(x) && requires { begin(*begin(x)); }) {
                cerr << '\n';
                m += 2;
                for (auto y : x)
                    cerr << string(m, ' ') << setw(2) << left << i++, p(y), cerr << '\n';
                cerr << string(m -= 2, ' ');
            } else  // normal iterable
                for (auto y : x) i++ ? C(',') : void(), p(y);
            C('}');
        } else if constexpr (requires { x.pop(); }) {  // stacks, queues
            C('{');
            while (!x.empty()) {
                if (i++) C(',');
                if constexpr (requires { x.top(); }) p(x.top());
                else p(x.front());
                x.pop();
            }
            C('}');
        } else if constexpr (requires { get<0>(x); }) {  // pairs, tuples
            C('(');
            apply([&](auto... args) { ((i++ ? C(',') : void(), p(args)), ...); }, x);
            C(')');
        } else C(x, 5);
    }
    template <typename T, typename... V>
    void printer(const char* names, T head, V ...tail) {
        int i = 0;
        for (int bs = 0; names[i] && (names[i] != ',' || bs); ++i)
            bs += !strchr(")>}", names[i]) - !strchr("(<{", names[i]);
        cerr.write(names, i), C(" = "), p(head);
        if constexpr (sizeof...(tail)) C(" |"), printer(names + i + 1, tail...);
        else cerr << '\n';
    }
}
#define dbg(...) DBG::C(__LINE__), DBG::C(": "), DBG::printer(#__VA_ARGS__, __VA_ARGS__)
```

# Algoritmos

## Geometria

### Ângulo entre segmentos

```c++
/**
 *  @param  P  First extreme point.
 *  @param  Q  Second extreme point.
 *  @param  R  First extreme point.
 *  @param  S  Second extreme point.
 *  @return    Smallest angle between segments PQ and RS in radians.
 *
 *  Time complexity: O(1)
*/
template <typename T>
double angle(const pair<T, T>& P, const pair<T, T>& Q,
             const pair<T, T>& R, const pair<T, T>& S) {
    T ux = P.x - Q.x, uy = P.y - Q.y;
    T vx = R.x - S.x, vy = R.y - S.y;
    T num = ux * vx + uy * vy;

    double den = hypot(ux, uy) * hypot(vx, vy);
    assert(den != 0.0);  // degenerate segment
    return acos(num / den);
}
```

### Distância entre pontos

```c++
/**
 *  @param  P  First point.
 *  @param  Q  Second point.
 *  @return    Distance between points P and Q.
 *
 *  Time complexity: O(1)
*/
template <typename T, typename S>
double dist(const pair<T, T>& P, const pair<S, S>& Q) {
    return hypot(P.x - Q.x, P.y - Q.y);
}
```

### Envoltório convexo

```c++
template <typename T>
vector<pair<T, T>> makeHull(const vector<pair<T, T>>& PS) {
    vector<pair<T, T>> hull;
    for (auto& P : PS) {
        ll sz = hull.size();  //           if want collinear < 0
        while (sz >= 2 and D(hull[sz - 2], hull[sz - 1], P) <= 0) {
            hull.pop_back();
            sz = hull.size();
        }
        hull.eb(P);
    }
    return hull;
}

/**
 *  @param  PS  Vector of points.
 *  @return     Convex hull.
 *
 *  Convex hull will be sorted counter-clockwise.
 *  First and last point will be the same.
 *
 *  Time complexity: O(Nlog(N))
*/
template <typename T>
vector<pair<T, T>> monotoneChain(vector<pair<T, T>> PS) {
    vector<pair<T, T>> lower, upper;
    sort(all(PS));
    lower = makeHull(PS);
    reverse(all(PS));
    upper = makeHull(PS);
    lower.pop_back();
    lower.emplace(lower.end(), all(upper));
    return lower;
}
```

### Orientação de ponto

```c++
/**
 *  @param  A  First extreme point.
 *  @param  B  Second extreme point.
 *  @param  P  Point.
 *  @return    Value that represents orientation of P to segment AB.
 *
 *  If orientation is collinear: zero;
 *  If point is to the left:     positive;
 *  If point is to the right:    negative;
 *
 *  Time complexity: O(1)
*/
template <typename T>
T D(const pair<T, T>& A, const pair<T, T>& B, const pair<T, T>& P) {
    return (A.x * B.y + A.y * P.x + B.x * P.y) - (P.x * B.y + P.y * A.x + B.x * A.y);
}
```

```c++
/**
 *  @param  P  First point.
 *  @param  Q  Second point.
 *  @param  O  Origin point.
 *  @return    True if P before Q in counter-clockwise order.
 *
 *  Time complexity: O(1)
*/
template <typename T>
bool ccw(pair<T, T> P, pair<T, T> Q, const pair<T, T>& O) {
    static const char qo[2][2] = { { 2, 3 }, { 1, 4 } };
    P.x -= O.x, P.y -= O.y, Q.x -= O.x, Q.y -= O.y, O.x = 0, O.y = 0;
    bool qqx = equals(P.x, 0) or P.x > 0, qqy = equals(P.y, 0) or P.y > 0;
    bool rqx = equals(Q.x, 0) or Q.x > 0, rqy = equals(Q.y, 0) or Q.y > 0;
    if (qqx != rqx || qqy != rqy) return qo[qqx][qqy] > qo[rqx][rqy];
    return equals(D(O, P, Q), 0) ?
           (P.x * P.x - P.y * P.y) < (Q.x * Q.x - Q.y * Q.y) : D(O, P, Q) > 0;
}
```

### Mediatriz

```c++
/**
 *  @param  P  First extreme point.
 *  @param  Q  Second extreme point.
 *  @return    Perpendicular bisector to segment PQ.
 *
 *  Time complexity: O(1)
*/
template <typename T>
Line<T> perpendicularBisector(const pair<T, T>& P, const pair<T, T>& Q) {
    T a = 2 * (Q.x - P.x), b = 2 * (Q.y - P.y);
    T c = (P.x * P.x + P.y * P.y) - (Q.x * Q.x + Q.y * Q.y);
    return { a, b, c };
}
```

### Rotação de ponto

```c++
/**
 *  @param  P  Point.
 *  @param  a  Angle in radians.
 *  @return    Rotated point.
 *
 *  Time complexity: O(1)
*/
template <typename T>
pd rotate(const pair<T, T>& P, double a) {
    double x = cos(a) * P.x - sin(a) * P.y;
    double y = sin(a) * P.x + cos(a) * P.y;
    return { x, y };
}
```

## Árvores

### Binary lifting

```c++
const ll LOG = 31;
vvll parent;
vll depth;

/**
 *  @brief     Binary lifting pre-processing.
 *  @param  g  Tree.
 *  @param  n  Quantity of nodes.
 *
 *  Time complexity: O(Vlog(V))
*/
void populate(const vvll& g, ll n) {
    parent.resize(n + 1, vll(LOG));
    depth.resize(n + 1);

    // initialize known relationships
    auto dfs = [&](auto&& self, ll u, ll p = 1) -> void {
        parent[u][0] = p;
        depth[u] = depth[p] + 1;
        for (ll v : g[u]) if (v != p)
            self(self, v, u);
    }; dfs(dfs, 1);

    rep(i, 1, LOG)
        rep(j, 1, n + 1)
            parent[j][i] = parent[ parent[j][i - 1] ][i - 1];
}

/**
 *  @param  u  Node.
 *  @param  k  Ancestor number, starts from 1.
 *  @return    k-th ancestor of u.
 *
 *  Requires that depth and parent are populated.
 *
 *  Time complexity: O(log(V))
*/
ll kthAncestor(ll u, ll k) {
    assert(k > 0 and u > 0 and u < parent.size());
    if (k > depth[u]) return -1;  // no kth ancestor
    rep(i, 0, LOG)
        if (k & (1LL << i))
            u = parent[u][i];
    return u;
}
```

### Centróide

```c++
vll subtree;

ll dfs(const vvll& g, ll u, ll p = 0) {
    if (subtree.empty()) subtree.resize(g.size());
    subtree[u] = 1;
    for (ll v : g[u]) if (v != p)
        subtree[u] += dfs(g, v, u);
    return subtree[u];
}

/**
 *  @param  g  Tree.
 *  @param  u  Root.
 *  @return    A new root that makes the size of all subtrees be n/2 or less.
 *
 *  Time complexity: O(E)
*/
ll centroid(const vvll& g, ll u, ll p = 0) {
    if (subtree.empty()) dfs(g, u, p);
    for (ll v : g[u]) if (v != p)
        if (subtree[v] * 2 > g.size() - 1)
            return centroid(g, v, u);
    return u;
}
```

### Euler Tour

```c++
ll timer = 0;
vll st, et;

/**
 *  @param  g  Tree.
 *  @param  u  Root.
 *
 *  Populates st and et, vectors that represents intervals of
 *  each subtree, with those we can use stuff like segtrees on
 *  the subtrees.
 *
 *  Time complexity: O(E)
*/
void eulerTour(const vvll& g, ll u, ll p = 0) {
    if (st.empty()) st.resize(g.size()), et.resize(g.size());
    assert(u < st.size());
    st[u] = timer++;
    for (ll v : g[u]) if (v != p)
        eulerTour(g, v, u);
    et[u] = timer++;
}
```

### Menor ancestral comum (LCA)

```c++
/**
 *  @param  u  First node.
 *  @param  v  Second node.
 *  @return    Lowest common ancestor between u and v.
 *
 *  Requires binary lifting pre-processing technique.
 *
 *  Time complexity: O(log(V))
*/
ll lca(ll u, ll v) {
    assert(u > 0 and v > 0 and u < parent.size() and v < parent.size());
    if (depth[u] < depth[v]) swap(u, v);
    ll k = depth[u] - depth[v];
    u = kthAncestor(u, k);
    if (u == v) return u;
    per(i, LOG - 1, 0)
        if (parent[u][i] != parent[v][i])
            u = parent[u][i], v = parent[v][i];
    return parent[u][0];  // could also be parent[v][0]
}
```

## Grafos

### Bellman-Ford

```c++
/**
 *  @param  g  Graph (w, v).
 *  @param  s  Starting vertex.
 *  @return    Vectors with smallest distances from every vertex to s and the paths.
 *
 *  Weights can be negative.
 *  Can detect negative cycles.
 *
 *  Time complexity: O(EV)
*/
pair<vll, vll> spfa(const vvpll& g, ll s) {
    vll ds(g.size(), LLONG_MAX), cnt(g.size()), pre = cnt;
    vb in_queue(g.size());
    queue<ll> q;
    ds[s] = 0; q.emplace(s);
    in_queue[s] = true;
    while (!q.empty()) {
        ll u = q.front(); q.pop();
        in_queue[u] = false;
        for (auto [w, v] : g[u]) {
            if (ds[u] == LLONG_MIN) {
                if (ds[v] != LLONG_MIN)
                    q.emplace(v);
                ds[v] = LLONG_MIN;
            }
            else if (ds[u] + w < ds[v]) {
                ds[v] = ds[u] + w;
                ++cnt[v], pre[v] = u;
                if (cnt[v] == g.size()) {
                    ds[v] = LLONG_MIN;
                    ds[0] = v;  // a node that has -inf dist
                }
                if (!in_queue[v]) {
                    q.emplace(v);
                    in_queue[v] = true;
                }
            }
        }
    }
    return { ds, pre };
}
```

### BFS 0/1

```c++
/**
 *  @param  g  Graph (w, v).
 *  @param  s  Starting vertex.
 *  @return    Vector with smallest distances from every vertex to s.
 *
 *  The graph can only have weights 0 and 1.
 *
 *  Time complexity: O(E)
*/
vll bfs01(const vvpll& g, ll s) {
    vll ds(g.size(), LLONG_MAX);
    deque<ll> dq;
    dq.eb(s); ds[s] = 0;
    while (!dq.empty()) {
        ll u = dq.front(); dq.pop_front();
        for (auto [w, v] : g[u])
            if (ds[u] + w < ds[v]) {
                ds[v] = ds[u] + w;
                if (w == 1) dq.eb(v);
                else dq.emplace_front(v);
            }
    }
    return ds;
}
```

### Caminho euleriano

```c++
/**
 *  @param  g  Graph.
 *  @param  d  Directed flag (true if g is directed).
 *  @param  s  Starting vertex.
 *  @param  e  Ending vertex.
 *  @return    Vector with the eulerian path. If e is specified: eulerian cycle.
 *
 *  Empty if impossible or no edges.
 *
 *  Time complexity: O(EVlog(EV))
*/
vll eulerianPath(const vvll& g, bool d, ll s, ll e = -1) {
    vector<multiset<ll>> h(g.size());
    vll res, in_degree(g.size());
    stack<ll> st;
    st.emplace(s);  // start vertex

    rep(u, 0, g.size())
        for (auto v : g[u]) {
            ++in_degree[v];
            h[u].emplace(v);
        }
    
    ll check = (in_degree[s] - (ll)h[s].size()) * (in_degree[e] - (ll)h[e].size());
    if (e != -1 and check != -1)
        return {};  // impossible
        
    rep(u, 0, h.size()) {
        if (e != -1 and (u == s or u == e)) continue;
        if (in_degree[u] != h[u].size() or (!d and in_degree[u] & 1))
            return {};  // impossible
    }

    while (!st.empty()) {
        ll u = st.top();
        if (h[u].empty()) {
            res.eb(u);
            st.pop();
        }
        else {
            ll v = *h[u].begin();
            h[u].erase(h[u].find(v));
            --in_degree[v];
            if (!d) {
                h[v].erase(h[v].find(u));
                --in_degree[u];
            }
            st.emplace(v);
        }
    }

    rep(u, 0, g.size())
        if (in_degree[u] != 0)
            return {};  // impossible

    reverse(all(res));
    return res;
}
```

### Dijkstra

```c++
/**
 *  @param  g  Graph (w, v).
 *  @param  s  Starting vertex.
 *  @return    Vectors with smallest distances from every vertex to s and the paths.
 *
 *  If want to calculate quantity of paths or size of path,
 *  notice that when the distance for a vertex is calculated
 *  it probably won't be the best, remember to reset calculations
 *  if a better is found.
 *
 *  Time complexity: O((Elog(V))
*/
pair<vll, vll> dijkstra(const vvpll& g, ll s) {
    vll ds(g.size(), LLONG_MAX), pre(g.size(), -1);
    priority_queue<pll, vpll, greater<>> pq;
    ds[s] = 0; pq.emplace(ds[s], s);
    while (!pq.empty()) {
        auto [t, u] = pq.top(); pq.pop();
        if (t > ds[u]) continue;
        for (auto [w, v] : g[u])
            if (t + w < ds[v]) {
                ds[v] = t + w, pre[v] = u;
                pq.emplace(ds[v], v);
            }
    }
    return { ds, pre };
}

vll getPath(const vll& pre, ll s, ll u) {
    vll p { u };
    do {
        p.eb(pre[u]);
        u = pre[u];
        if (u == 0) return {};
    } while (u != s);
    reverse(all(p));
    return p;
}
```

### Floyd Warshall

```c++
/**
 *  @param  g  Graph (w, v).
 *  @return    Vector with smallest distances between every vertex.
 *
 *  Weights can be negative.
 *  Can detect negative cycles.
 *
 *  Time complexity: O(V^3)
*/
vvll floydWarshall(const vvpll& g) {
    ll n = g.size();
    vvll ds(n + 1, vll(n + 1, INT_MAX));

    rep(u, 1, n) {
        ds[u][u] = 0;
        for (auto [w, v] : g[u]) {
            ds[u][v] = min(ds[u][v], w);
            if (ds[u][u] < 0) ds[u][u] = INT_MIN;  // negative cycle
        }
    }

    rep(k, 1, n) rep(u, 1, n) rep(v, 1, n)
        if (ds[u][k] != INT_MAX and ds[k][v] != INT_MAX) {
            if (ds[k][k] == INT_MIN) ds[u][v] = INT_MIN;
            else {
                ds[u][v] = min(ds[u][v], ds[u][k] + ds[k][v]);
                if (ds[u][v] < 0) ds[u][v] = INT_MIN;
            }
        }

    return ds;
}
```

### Kosaraju

```c++
/**
 *  @param  g  Directed graph.
 *  @return    Condensed graph and strongly connected components.
 *
 *  Time complexity: O((Elog(V))
*/
pair<vvll, map<ll, vll>> kosaraju(const vvll& g) {
    vvll g_inv(g.size()), g_cond(g.size());
    map<ll, vll> scc;
    vb vs(g.size());
    vll order, reprs(g.size());

    auto dfs = [&vs](auto&& self, const vvll& h, vll& out, ll u) -> ll {
        ll repr = u;
        vs[u] = true;
        for (ll v : h[u]) if (!vs[v])
            repr = min(repr, self(self, h, out, v));
        out.eb(u);
        return repr;
    };

    rep(u, 1, g.size()) {
        for (ll v : g[u])
            g_inv[v].eb(u);
        if (!vs[u])
            dfs(dfs, g, order, u);
    }

    vs.assign(g.size(), false);
    reverse(all(order));

    for (ll u : order)
        if (!vs[u]) {
            vll cc;
            ll repr = dfs(dfs, g_inv, cc, u);
            scc[repr] = cc;
            for (ll v : cc)
                reprs[v] = repr;
        }

    rep(u, 1, g.size())
        for (ll v : g[u])
            if (reprs[u] != reprs[v])
                g_cond[reprs[u]].eb(reprs[v]);

    return { g_cond, scc };
}
```

### Kruskal

```c++
/**
 *  @brief         Get min/max spanning tree.
 *  @param  edges  Vector of edges (w, u, v).
 *  @param  n      Quantity of vertex.
 *  @return        Edges of mst, or forest if not connected and sum of weights.
 *
 *  Time complexity: O(Elog(E))
*/
pair<vtll, ll> kruskal(vtll& edges, ll n) {
    DSU dsu(n);
    vtll mst;
    ll edges_sum = 0;
    sort(all(edges));  // change order if want maximum
    for (auto [w, u, v] : edges) if (!dsu.sameSet(u, v)) {
        dsu.mergeSetsOf(u, v);
        mst.eb(w, u, v);
        edges_sum += w;
    }
    return { mst, edges_sum };
}
```

### Ordenação topológica

```c++
/**
 *  @param  g  Directed graph.
 *  @return    Vector with vertexes in topological order or empty if has cycle.
 *
 *  Time complexity: O(EVlog(V))
*/
vll topologicalSort(const vvll& g) {
    vll degree(g.size()), res;
    rep(u, 1, g.size())
        for (ll v : g[u])
            ++degree[v];

    // lower values bigger priorities
    priority_queue<ll, vll, greater<>> pq;
    rep(u, 1, degree.size())
        if (degree[u] == 0)
            pq.emplace(u);

    while (!pq.empty()) {
        ll u = pq.top();
        pq.pop();
        res.eb(u);
        for (ll v : g[u])
            if (--degree[v] == 0)
                pq.emplace(v);
    }

    if (res.size() != g.size() - 1) return {};  // cycle
    return res;
}
```

### Max flow/min cut (Dinic)

```c++
/**
 *  @param  g  Graph (w, v).
 *  @param  s  Source.
 *  @param  t  Sink.
 *  @return    Max flow/min cut and graph with residuals.
 *
 *  If want the cut edges do a dfs, after, for every visited
 *  node if it has edge to v but this is not visited then it
 *  was a cut.
 *
 *  If want all the paths from source to sink, make a bfs,
 *  only traverse if there is a path from u to v and w is 0.
 *  When getting the path set each w in the path to 1.
 *
 *  Time complexity: O(EV^2) but there is cases
 *                   where it's better.
*/
pair<ll, vector<vtll>> maxFlow(const vvpll& g, ll s, ll t) {
    ll n = g.size();
    vector<vtll> h(n);  // (w, v, rev)
    vll lvl(n), ptr(n), q(n);
    
    rep(u, 1, n)
        for (auto [w, v] : g[u]) {
            h[u].eb(w, v, h[v].size());
            h[v].eb(0, u, h[u].size() - 1);
        }

    auto dfs = [&](auto&& self, ll u, ll nf) -> ll {
        if (u == t or nf == 0) return nf;
        for (ll& i = ptr[u]; i < h[u].size(); i++) {
            auto& [w, v, rev] = h[u][i];
            if (lvl[v] == lvl[u] + 1)
                if (ll p = self(self, v, min(nf, w))) {
                    auto& [wv, _, __] = h[v][rev];
                    w -= p, wv += p;
                    return p;
                }
        }
        return 0;
    };
    
    ll f = 0;
    q[0] = s;
    
    rep(l, 0, 31)
        do {
            lvl = ptr = vll(n);
            ll qi = 0, qe = lvl[s] = 1;
            while (qi < qe and !lvl[t]) {
                ll u = q[qi++];
                for (auto [w, v, rev] : h[u])
                    if (!lvl[v] and w >> (30 - l))
                        q[qe++] = v, lvl[v] = lvl[u] + 1;
            }

            while (ll nf = dfs(dfs, s, LLONG_MAX))
                f += nf;
        } while (lvl[t]);
        
    return { f, h };
}
```

## Outros

### Maior subsequência comum (LCS)

```c++
/**
 *  @param  xs  First vector.
 *  @param  ys  Second vector.
 *  @return     Size of longest common subsequence.
 *
 *  Time complexity: O(N^2)
*/
template <typename T>
ll lcs(const T& xs, const T& ys) {
    vvll dp(xs.size() + 1, vll(ys.size() + 1));
    rep(i, 1, xs.size() + 1)
        rep(j, 1, ys.size() + 1)
            if (xs[i - 1] == ys[j - 1])
                dp[i][j] = 1 + dp[i - 1][j - 1];
            else
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j]);
    return dp.back().back();
}
```

### Maior subsequência crescente (LIS)

```c++
/**
 *  @param  xs      Vector.
 *  @param  values  True if want values, indexes otherwise.
 *  @return         Longest increasing subsequence as values or indexes.
 *
 *  Time complexity: O(Nlog(N))
*/
vll lis(const vll& xs, bool values) {
    assert(!xs.empty());
    vll ss, idx, pre(xs.size());
    rep(i, 0, xs.size()) {
        // change to upper_bound if want not decreasing
        ll j = lower_bound(all(ss), xs[i]) - ss.begin();
        if (j == ss.size()) idx.eb(), ss.eb();
        if (j == 0) pre[i] = -1;
        else        pre[i] = idx[j - 1];
        idx[j] = i;
        ss[j]  = xs[i];
    }
    vll ys;
    ll i = idx.back();
    while (i != -1) {
        ys.eb((values ? xs[i] : i));
        i = pre[i];
    }
    reverse(all(ys));
    return ys;
}
```

## Matemática

### Coeficiente binomial

```c++
/**
 *  @param  n  First number.
 *  @param  k  Second number.
 *  @return    Binomial coefficient.
 *
 *  Time complexity: O(N^2)/O(1)
*/
ll binom(ll n, ll k) {
    const ll MAXN = 64 + 2;
    static vvll dp(MAXN + 1, vll(MAXN + 1));
    if (dp[0][0] != 1) {
        dp[0][0] = 1;
        for (ll i = 1; i <= MAXN; i++)
            for (ll j = 0; j <= i; j++)
                dp[i][j] = dp[i - 1][j] + (j ? (dp[i - 1][j - 1]) : 0);
    }
    if (n < k or n * k < 0) return 0;
    return dp[n][k];
}
```

### Coeficiente binomial mod

```c++
/**
 *  @param  n  First number.
 *  @param  k  Second number.
 *  @return    Binomial coefficient mod M.
 *
 *  Time complexity: O(N)/O(1)
*/
ll binom(ll n, ll k) {
    const ll MAXN = 3e6, M = 1e9 + 7;  // check mod value!
    static vll fac(MAXN + 1), inv(MAXN + 1), finv(MAXN + 1);
    if (fac[0] != 1) {
        fac[0] = fac[1] = inv[1] = finv[0] = finv[1] = 1;
        rep(i, 2, MAXN + 1) {
            fac[i] = fac[i - 1] * i % M;
            inv[i] = M - M / i * inv[M % i] % M;
            finv[i] = finv[i - 1] * inv[i] % M;
        }
    }
    if (n < k or n * k < 0) return 0;
    return fac[n] * finv[k] % M * finv[n - k] % M;
}
```

### Conversão de base

```c++
/**
 *  @param  x  Number in base 10.
 *  @param  b  Base.
 *  @return    Vector with coefficients of x in base b.
 *
 *  Example: (x = 6, b = 2): { 1, 1, 0 }
 *
 *  Time complexity: O(log(N))
*/
vll toBase(ll x, ll b) {
    assert(b != 0);
    vll res;
    while (x) {
        res.eb(x % b);
        x /= b;
    }
    reverse(all(res));
    return res;
}
```

### Crivo de Eratóstenes

```c++
/**
 *  @param  n  Bound.
 *  @return    Vectors with primes from [1, n] and smallest prime factors.
 *
 *  Time complexity: O(Nlog(N))
*/
pair<vll, vll> sieve(ll n) {
    vll ps, spf(n + 1);
    rep(i, 2, n + 1) if (!spf[i]) {
        ps.eb(i);
        for (ll j = i; j <= n; j += i)
            if (!spf[j]) spf[j] = i;
    }
    return { ps, spf };
}
```

### Divisores

```c++
/**
 *  @param  x  Target.
 *  @return    Unordered vector with all divisors of x.
 *
 *  Time complexity: O(sqrt(N))
*/
vll divisors(ll x) {
    vll ds;
    for (ll i = 1; i * i <= x; ++i)
        if (x % i == 0) {
            ds.eb(i);
            if (i * i != x) ds.eb(x / i);
        }
    return ds;
}
```

### Divisores de vários números

```c++
/**
 *  @param  xs  Target vector.
 *  @param  x   Number.
 *  @return     Divisors of x.
 *
 *  Time complexity: O(Nlog(N))
*/
vll divisors(const vll& xs, ll x) {
    static ll MAXN = 1e6 + 2;
    static vll hist(MAXN);
    static vvll ds(MAXN);
    if (MAXN == 1e6 + 2) {
        MAXN = 0;
        for (ll y : xs) {
            assert(y <= 1e6);
            MAXN = max(MAXN, y + 1);
            ++hist[y];
        }
        
        rep(i, 1, MAXN)
            for (ll j = i; j < MAXN; j += i)
                if (hist[j]) ds[j].eb(i);
    }
    return ds[x];
}
``

### Fatoração

```c++
/**
 *  @param  x  Target.
 *  @return    Vector with all prime factors of x.
 *
 *  Time complexity: O(sqrt(N))
*/
vll factors(ll x) {
    vll fs;
    for (ll i = 2; i * i <= x; ++i)
        while (x % i == 0) {
            fs.eb(i);
            x /= i;
        }
    if (x > 1) fs.eb(x);
    return fs;
}
```

### Fatoração com crivo

```c++
/**
 *  @param  x    Target.
 *  @param  spf  Vector of smallest prime factors
 *  @return      Vector with all prime factors of x.
 *
 *  Requires sieve.
 *
 *  Time complexity: O(log(N))
*/
vll factors(ll x, const vll& spf) {
    vll fs;
    while (x != 1) {
        fs.eb(spf[x]);
        x /= spf[x];
    }
    return fs;
}
```

### Quantidade de divisores

```c++
/**
 *  @param  x  Target.
 *  @return    Quantity of divisors of x.
 *
 *  Time complexity: O(Nlog(N))/O(1)
*/
ll qntDivisors(ll x) {
    const ll MAXN = 1e6;
    static vll qnt(MAXN + 1);
    if (qnt[1] != 1)
        rep(i, 1, MAXN + 1)
            for (ll j = i; j <= MAXN; j += i)
                ++qnt[j];
    assert(x >= 0 and x <= MAXN);
    return qnt[x];
}
```

### Permutação com repetição

```c++
/**
 *  @param  hist  Histogram.
 *  @return       Permutation with repetition mod M.
 *
 *  Time complexity: O(N)
*/
template <typename T>
ll rePerm(const map<T, ll>& hist) {
    const ll MAXN = 3e6, M = 1e9 + 7;  // check mod value!
    static vll fac(MAXN + 1), inv(MAXN + 1), finv(MAXN + 1);
    if (fac[0] != 1) {
        fac[0] = fac[1] = inv[1] = finv[0] = finv[1] = 1;
        rep(i, 2, MAXN + 1) {
            fac[i] = fac[i - 1] * i % M;
            inv[i] = M - M / i * inv[M % i] % M;
            finv[i] = finv[i - 1] * inv[i] % M;
        }
    }
    if (hist.empty()) return 0;
    ll res = 1, total = 0;
    for (auto [k, v] : hist) {
        res = res * finv[v] % M;
        total += v;
    }
    return res * fac[total] % M;
}
```

## Strings

### Distância de edição

```c++
/**
 *  @param  s  First string.
 *  @param  t  Second string.
 *  @return    Edit distance to transform s in t and operations.
 *
 *  Can change costs.
 *  -      Deletion
 *  c      Insertion of c
 *  =      Keep
 *  [c->d] Substitute c to d.
 *
 *  Time complexity: O(MN)
*/
pair<ll, string> edit(const string& s,  string& t) {
    ll ci = 1, cr = 1, cs = 1, m = s.size(), n = t.size();
    vvll dp(m + 1, vll(n + 1));
    vvll pre = dp;

    rep(i, 0, m + 1)
        dp[i][0] = i*cr, pre[i][0] = 'r';

    rep(j, 0, n + 1)
        dp[0][j] = j*ci, pre[0][j] = 'i';

    rep(i, 1, m + 1)
        rep(j, 1, n + 1) {
            ll ins = dp[i][j - 1] + ci, del = dp[i - 1][j] + cr;
            ll subs = dp[i - 1][j - 1] + cs * (s[i - 1] != t[j - 1]);
            dp[i][j] = min({ ins, del, subs });
            pre[i][j] = (dp[i][j] == ins ? 'i' : (dp[i][j] == del ? 'r' : 's'));
        }

    ll i = m, j = n;
    string ops;

    while (i or j) {
        if (pre[i][j] == 'i')
            ops += t[--j];
        else if (pre[i][j] == 'r') {
            ops += '-';
            --i;
        }
        else {
            --i, --j;
            if (s[i] == t[j])
                ops += '=';
            else
                ops += "]", ops += t[j], ops += ">-", ops += s[i], ops += "[";
        }
    }
    
    reverse(all(ops));
    return { dp[m][n], ops };
}
```

### Z-Function

```c++
/**
 *  @param  s  String.
 *  @return    Vector with Z-Function value for every position.
 *
 *  Time complexity: O(N)
*/
vll z(const string& s) {
    ll n = s.size(), l = 0, r = 0;
    vll zs(n);

    rep(i, 1, s.size()) {
        if (i <= r)
            zs[i] = min(zs[i - l], r - i + 1);

        while (zs[i] + i < n && s[zs[i]] == s[i + zs[i]])
            ++zs[i];

        if (r < i + zs[i] - 1)
            l = i, r = i + zs[i] - 1;
    }

    return zs;
}
```

### Ocorrências de substring

```c++
/**
 *  @param  s  String.
 *  @param  t  Substring.
 *  @return    Vector with the first index of occurrences.
 *
 *  Requires Z-Function.
 *
 *  Time complexity: O(N)
*/
vll occur(const string& s, const string& t) {
    auto zs = z(t + ';' + s);
    vll is;
    rep(i, 0, zs.size())
        if (zs[i] == t.size())
            is.eb(i - t.size() - 1);
    return is;
}
```

# Estruturas

## Árvores

### BIT tree 2D

```c++
/**
 *  @brief Make rectangular interval sum queries and point updates.
*/
template <typename T>
struct BIT2D {
    /**
     *  @param  h  Height.
     *  @param  w  Width.
    */
    BIT2D(ll h, ll w) : n(h), m(w), bit(n + 1, vector<T>(m + 1)) {}
    
    /**
     *  @brief     Adds v to position (y, x).
     *  @param  y  First position.
     *  @param  x  Second position.
     *  @param  v  Value to add.
     *
     *  1-indexed
     *
     *  Time complexity: O(log(N))
    */
    void add(ll y, ll x, T v) {
        assert(y > 0 and x > 0 and y <= n and x <= m)
       	for (; y <= n; y += y & -y)
            for (ll i = x; i <= m; i += i & -i)
                bit[y][i] += v;
    }
    
    T sum(ll y, ll x) {
        assert(y > 0 and x > 0 and y <= n and x <= m)
        T sum = 0;
        for (; y > 0; y -= y & -y)
            for (ll i = x; i > 0; i -= i & -i)
                sum += bit[y][i];
        return sum;
    }
    
    /**
     *  @param  ly  Lower y bound.
     *  @param  lx  Lower x bound.
     *  @param  hy  Higher y bound.
     *  @param  hx  Higher x bound.
     *  @return     Sum in that rectangle.
     *
     *  1-indexed
     *
     *  Time complexity: O(log(N))
    */
    T sum(ll ly, ll lx, ll hy, ll hx) {
        return sum(hy, hx)     - sum(hy, lx - 1) -
               sum(ly - 1, hx) + sum(ly - 1, lx - 1);
    }
    
    ll n, m;
    vector<vector<T>> bit;
};
```

### Disjoint set union

```c++
struct DSU {
    /**
     *  @param  n  Size.
    */
    DSU(ll n) : parent(n + 1), size(n + 1, 1) {
        iota(all(parent), 0);
    }

    /**
    *  @param  x  Element.
    *
    *  Time complexity: ~O(1)
    */
    ll setOf(ll x) {
        assert(x >= 0 and x < parent.size());
        return parent[x] == x ? x : parent[x] = setOf(parent[x]);
    }

    /**
    *  @param  x  Element.
    *  @param  y  Element.
    *
    *  Time complexity: ~O(1)
    */
    void mergeSetsOf(ll x, ll y) {
        ll a = setOf(x), b = setOf(y);
        if (size[a] > size[b]) swap(a, b);
        parent[a] = b;
        if (a != b) size[b] += size[a];
    }

    /**
    *  @param  x  Element.
    *  @param  y  Element.
    *
    *  Time complexity: ~O(1)
    */
    bool sameSet(ll x, ll y) { return setOf(x) == setOf(y); }

    vll parent, size;
};
```

### Red-Black tree (ordered set)

```c++
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace __gnu_pbds;

/**
*  @brief Like a multiset but with indexes.
*/
template <typename T>
struct RBT {
    void insert(T x) { rb.insert({ x, n++ }); }
    ll size() { return rb.size(); };
    bool empty() { return rb.empty(); };
        
    void erase(T x) {
        auto it = rb.lower_bound({ x, 0 });
        if (it == rb.end() or it->first != x) return;
        rb.erase(it);
    }
    
    /**
    *  @param  x  Element.
    *  @return    Index of where x would be if inserted.
    *
    *  Can also interpret as quantity of elements smaller than x.
    *
    *  Time complexity: O(log(N))
    */
    ll order_of_key(T x) { return rb.order_of_key({ x, 0 }); }
    
    /**
    *  @param  i  Index.
    *  @return    Element at index i.
    *
    *  Time complexity: O(log(N))
    */
    T find_by_order(ll i) { 
        auto it = rb.find_by_order(i);
        if (it == rb.end()) return -1;  // value for not found
        return it->first;
    }

    ll n = 1;
    tree<pair<T, ll>, null_type, less<>,
    rb_tree_tag, tree_order_statistics_node_update> rb;
};
```

```c++
/**
*  @brief Like a set but with indexes.
*/
template <typename T>
using RBT = tree<T, null_type, less<>,
rb_tree_tag, tree_order_statistics_node_update>;
```

### Segment tree

```c++
/**
*  @brief Make interval queries and updates
*/
template <typename T, typename Op = function<T(T, T)>>
struct Segtree {
    /**
    *  @param  sz   Size.
    *  @param  f    Merge function.
    *  @param  def  Default value.
    *
    *  Example: def in sum or gcd should be 0, in max LLONG_MIN, in min LLONG_MAX
    */
    Segtree(ll sz, Op f, T def)
        : seg(4 * sz, def), lzy(4 * sz), n(sz), op(f), DEF(def) {}

    /**
    *  @param  i  First interval extreme.
    *  @param  j  Second interval extreme.
    *  @param  x  Value to add (if it's a query).
    *  @return    f of [i, j] (if it's a query).
    *
    *  It's a query if x is specified.
    *
    *  Time complexity: O(log(N))
    */
    T setQuery(ll i, ll j, T x = LLONG_MIN, ll l = 0, ll r = -1, ll no = 1) {
        assert(i >= 0 and j >= 0 and i < n and j < n);
        if (r == -1) r = n - 1;
        if (lzy[no]) unlazy(l, r, no);
        if (j < l or i > r) return DEF;
        if (i <= l and r <= j) {
            if (x != LLONG_MIN) { 
                lzy[no] += x;
                unlazy(l, r, no);
            }
            return seg[no]; 
        }
        ll m = (l + r) / 2;
        T q = op(setQuery(i, j, x, l, m, 2 * no),
                 setQuery(i, j, x, m + 1, r, 2 * no + 1));
        seg[no] = op(seg[2 * no], seg[2 * no + 1]); 
        return q;                                         
    }

private:
    void unlazy(ll l, ll r, ll no) {
        if (seg[no] == DEF) seg[no] = 0;
        seg[no] += (r - l + 1) * lzy[no];  // sum
        // seg[no] += lzy[no];  // min/max
        if (l < r) {
            lzy[2 * no] += lzy[no];
            lzy[2 * no + 1] += lzy[no];
        }
        lzy[no] = 0;
    }

    vector<T> seg, lzy;
    ll n;
    Op op;
    T DEF = {};
};
```

### 3 Maiores valores

```c++
// not tested
struct MX3 {
    ll first, second, third;
    MX3(ll a, ll b, ll c) : first(a), second(b), third(c) {}
    MX3(ll x)             : first(x), second(x), third(x) {}
    MX3() = default;
    MX3 operator+=(MX3 other) {
        auto [a, b, c] = other;
        first += a, second += a, third += a;
        return *this;
    }
    bool operator!=(ll x) { return first != x; }
    operator bool() { return first; }
};

MX3 f(MX3 a, MX3 b) {
    vll xs { a.first, a.second, a.third, b.first, b.second, b.third };
    sort(all(xs), greater<>());
    xs.erase(unique(all(xs)), xs.end());
    return { xs[0], (xs.size() > 1 ? xs[1] : 0), (xs.size() > 2 ? xs[2] : 0) };
};
```

### Wavelet Tree

```c++
struct WaveletTree {
    /**
    *  @param  xs  Compressed vector.
    *  @param  n   Distinct elements amount in xs.
    *
    *  Sorts xs in the process.
    *
    *  Time complexity: O(Nlog(N))
    */
    WaveletTree(vll& xs, ll n) : wav(2 * n), n(n) {
        auto build = [&](auto&& self, auto b, auto e, ll l, ll r, ll no) {
            if (l == r) return;
            ll m = (l + r) / 2, i = 0;
            wav[no].resize(e - b + 1);
            for (auto it = b; it != e; ++it, ++i)
                wav[no][i + 1] = wav[no][i] + (*it <= m);
            auto p = stable_partition(b, e, [m](ll i) { return i <= m; });
            self(self, b, p, l, m, 2 * no);
            self(self, p, e, m + 1, r, 2 * no + 1);
        };
        build(build, all(xs), 0, n - 1, 1);
    }

    /**
    *  @param  i  First interval extreme.
    *  @param  j  Second interval extreme.
    *  @param  k  Value, starts from 1.
    *  @return    k-th smallest element in [i, j].
    *
    *  Time complexity: O(log(N))
    */
    ll kTh(ll i, ll j, ll k) {
        ++j;
        ll l = 0, r = n - 1, no = 1;
        while (l != r) {
            ll m = (l + r) / 2;
            ll seqm_l = wav[no][i], seqm_r = wav[no][j];
            no *= 2;
            if (k <= seqm_r - seqm_l)
                i = seqm_l, j = seqm_r, r = m;
            else
                k -= seqm_r - seqm_l, i -= seqm_l, j -= seqm_r, l = m + 1, ++no;
        }
        return l;
    }

    vvll wav;
    ll n;
};
```

## Geometria

### Reta

```c++
/**
*  A line with normalized coefficients.
*/
template <typename T>
struct Line {
    /**
    *  @param  P  First point.
    *  @param  Q  Second point.
    *
    *  Time complexity: O(1)
    */
    Line(const pair<T, T>& P, const pair<T, T>& Q)
            : a(P.y - Q.y), b(Q.x - P.x), c(P.x * Q.y - Q.x * P.y) {
        if constexpr (is_floating_point_v<T>) b /= a, c /= a, a = 1;
        else {
            if (a < 0 or (a == 0 and b < 0)) a *= -1, b *= -1, c *= -1;
            T gcd_abc = gcd(a, gcd(b, c));
            a /= gcd_abc, b /= gcd_abc, c /= gcd_abc;
        }
    }

    /**
    *  @param  P  Point.
    *  @return    True if P is in this line.
    *
    *  Time complexity: O(1)
    */
    bool contains(const pair<T, T>& P) { return equals(a * P.x + b * P.y + c, 0); }

    /**
    *  @param  r  Line.
    *  @return    True if r is parallel to this line.
    *
    *  Time complexity: O(1)
    */
    bool parallel(const Line& r) {
        T det = a * r.b - b * r.a;
        return equals(det, 0);
    }

    /**
    *  @param  r  Line.
    *  @return    True if r is orthogonal to this line.
    *
    *  Time complexity: O(1)
    */
    bool orthogonal(const Line& r) { return equals(a * r.a + b * r.b, 0); }

    /**
    *  @param  r  Line.
    *  @return    Point of intersection between r and this line.
    *
    *  Time complexity: O(1)
    */
    pd intersection(const Line& r) {
        double det = r.a * b - r.b * a;

        // same or parallel
        if (equals(det, 0)) return {};

        double x = (-r.c * b + c * r.b) / det;
        double y = (-c * r.a + r.c * a) / det;
        return { x, y };
    }

    /**
    *  @param  P  Point.
    *  @return    Distance from P to this line.
    *
    *  Time complexity: O(1)
    */
    double dist(const pair<T, T>& P) {
        return abs(a * P.x + b * P.y + c) / hypot(a, b);
    }

    /**
    *  @param  P  Point.
    *  @return    Closest point in this line to P.
    *
    *  Time complexity: O(1)
    */
    pd closest(const pair<T, T>& P) {
        double den = a * a + b * b;
        double x = (b * (b * P.x - a * P.y) - a * c) / den;
        double y = (a * (-b * P.x + a * P.y) - b * c) / den;
        return { x, y };
    }

    bool operator==(const Line& r) {
        return equals(a, r.a) and equals(b, r.b) and equals(c, r.c);
    }

    T a, b, c;
};
```

### Segmento

```c++
template <typename T>
struct Segment {
    /**
    *  @param  P  First extreme point.
    *  @param  Q  Second extreme point.
    */
    Segment(const pair<T, T>& P, const pair<T, T>& Q) : A(P), B(Q) {}

    /**
    *  @param  P  Point.
    *  @return    True if P is in this segment.
    *
    *  Time complexity: O(1)
    */
    bool contains(const pair<T, T>& P) const {
        T xmin = min(A.x, B.x), xmax = max(A.x, B.x);
        T ymin = min(A.y, B.y), ymax = max(A.y, B.y);
        if (P.x < xmin || P.x > xmax || P.y < ymin || P.y > ymax) return false;
        return equals((P.y - A.y) * (B.x - A.x), (P.x - A.x) * (B.y - A.y));
    }

    /**
    *  @param  r  Segment.
    *  @return    True if r intersects with this segment.
    *
    *  Time complexity: O(1)
    */
    bool intersect(const Segment& r) {
        T d1 = D(A, B, r.A),  d2 = D(A, B, r.B);
        T d3 = D(r.A, r.B, A), d4 = D(r.A, r.B, B);
        d1 /= d1 ? abs(d1) : 1, d2 /= d2 ? abs(d2) : 1;
        d3 /= d3 ? abs(d3) : 1, d4 /= d4 ? abs(d4) : 1;

        if ((equals(d1, 0) and contains(r.A)) or (equals(d2, 0) and contains(r.B)))
            return true;

        if ((equals(d3, 0) and r.contains(A)) or (equals(d4, 0) and r.contains(B)))
            return true;

        return (d1 * d2 < 0) and (d3 * d4 < 0);
    }

    /**
    *  @param  P  Point.
    *  @return    Closest point in this segment to P.
    *
    *  Time complexity: O(1)
    */
    pair<T, T> closest(const pair<T, T>& P) {
        Line<T> r(A, B);
        pd Q = r.closest(P);
        double distA = dist(A, P), distB = dist(B, P);
        if (this->contains(Q)) return Q;
        if (distA <= distB) return A;
        return B;
    }

    pair<T, T> A, B;
};
```

### Círculo

```c++
enum Position { IN, ON, OUT };

template <typename T>
struct Circle {
    /**
    *  @param  P  Origin point.
    *  @param  r  Radius length.
    */
    Circle(const pair<T, T>& P, T r) : C(P), r(r) {}

    /**
    *  Time complexity: O(1)
    */
    double area() { return acos(-1.0) * r * r; }
    double perimeter() { return 2.0 * acos(-1.0) * r; }
    double arc(double radians) { return radians * r; }
    double chord(double radians) { return 2.0 * r * sin(radians / 2.0); }
    double sector(double radians) { return (radians * r * r) / 2.0; }

    /**
    *  @param  a  Angle in radians.
    *  @return    Circle segment.
    *
    *  Time complexity: O(1)
    */
    double segment(double a) {
        double c = chord(a);
        double s = (r + r + c) / 2.0;
        double t = sqrt(s) * sqrt(s - r) * sqrt(s - r) * sqrt(s - c);
        return sector(a) - t;
    }

    /**
    *  @param  P  Point.
    *  @return    Value that represents orientation of P to this circle.
    *
    *  Time complexity: O(1)
    */
    Position position(const pair<T, T>& P) {
        double d = dist(P, C);
        return equals(d, r) ? ON : (d < r ? IN : OUT);
    }

    /**
    *  @param  c  Circle.
    *  @return    Intersection/s point between c and this circle.
    *
    *  Time complexity: O(1)
    */
    vector<pair<T, T>> intersection(const Circle& c) {
        double d = dist(c.C, C);

        // no intersection or same
        if (d > c.r + r or d < abs(c.r - r) or (equals(d, 0) and equals(c.r, r)))
            return {};

        double a = (c.r * c.r - r * r + d * d) / (2.0 * d);
        double h = sqrt(c.r * c.r - a * a);
        double x = c.C.x + (a / d) * (C.x - c.C.x);
        double y = c.C.y + (a / d) * (C.y - c.C.y);
        pd p1, p2;
        p1.x = x + (h / d) * (C.y - c.C.y);
        p1.y = y - (h / d) * (C.x - c.C.x);
        p2.x = x - (h / d) * (C.y - c.C.y);
        p2.y = y + (h / d) * (C.x - c.C.x);
        return p1 == p2 ? vector<pair<T, T>> { p1 } : vector<pair<T, T>> { p1, p2 };
    }

    /**
    *  @param  P  First point
    *  @param  Q  Second point
    *  @return    Intersection point/s between line PQ and this circle.
    *
    *  Time complexity: O(1)
    */
    vector<pd> intersection(pair<T, T> P, pair<T, T> Q) {
        P.x -= C.x, P.y -= C.y, Q.x -= C.x, Q.y -= C.y;
        double a(P.y - Q.y), b(Q.x - P.x), c(P.x * Q.y - Q.x * P.y);
        double x0 = -a * c / (a * a + b * b), y0 = -b * c / (a * a + b * b);
        if (c*c > r*r * (a*a + b*b) + 1e-9) return {};
        if (equals(c*c, r*r * (a*a + b*b))) return { { x0, y0 } };
        double d = r * r - c * c / (a * a + b * b);
        double mult = sqrt(d / (a * a + b * b));
        double ax = x0 + b * mult + C.x;
        double bx = x0 - b * mult + C.x;
        double ay = y0 - a * mult + C.y;
        double by = y0 + a * mult + C.y;
        return { { ax, ay }, { bx, by } };
    }

    /**
    *  @return Tangent points looking from origin.
    *
    *  Time complexity: O(1)
    */
    pair<pd, pd> tanPoints() {
        double b = hypot(C.x, C.y), th = acos(r / b);
        double d = atan2(-C.y, -C.x), d1 = d + th, d2 = d - th;
        return { {C.x + r * cos(d1), C.y + r * sin(d1)},
                 {C.x + r * cos(d2), C.y + r * sin(d2)} };
    }

    /**
    *  @param  P  First point
    *  @param  Q  Second point
    *  @param  R  Third point
    *  @return Circle defined by those 3 points.
    *
    *  Time complexity: O(1)
    */
    static Circle<double> from3(const pair<T, T>& P, const pair<T, T>& Q,
                                                     const pair<T, T>& R) {
        T a = 2 * (Q.x - P.x), b = 2 * (Q.y - P.y);
        T c = 2 * (R.x - P.x), d = 2 * (R.y - P.y);
        double det = a * d - b * c;

        // collinear points
        if (equals(det, 0)) return { { 0, 0 }, 0 };

        T k1 = (Q.x * Q.x + Q.y * Q.y) - (P.x * P.x + P.y * P.y);
        T k2 = (R.x * R.x + R.y * R.y) - (P.x * P.x + P.y * P.y);
        double cx = (k1 * d - k2 * b) / det;
        double cy = (a * k2 - c * k1) / det;
        return { { cx, cy }, dist(P, { cx, cy }) };
    }

    /**
    *  @param  PS  Points
    *  @return     Minimum enclosing circle with those points.
    *
    *  Time complexity: O(N)
    */
    static Circle<double> mec(vector<pair<T, T>>& PS) {
        random_shuffle(all(PS));
        Circle<double> c(PS[0], 0);
        rep(i, 0, PS.size()) {
            if (c.position(PS[i]) != OUT) continue;
            c = { PS[i], 0 };
            rep(j, 0, i) {
                if (c.position(PS[j]) != OUT) continue;
                c = {
                    { (PS[i].x + PS[j].x) / 2.0, (PS[i].y + PS[j].y) / 2.0 },
                       dist(PS[i], PS[j]) / 2.0
                };
                rep(k, 0, j)
                    if (c.position(PS[k]) == OUT)
                    c = from3(PS[i], PS[j], PS[k]);
            }
        }
        return c;
    }

    pair<T, T> C;
    T r;
};
```

### Triângulo

```c++
enum Class { EQUILATERAL, ISOSCELES, SCALENE };
enum Angles { RIGHT, ACUTE, OBTUSE };

template <typename T>
struct Triangle {
    /**
    *  @param  P  First point.
    *  @param  Q  Second point.
    *  @param  r  Third point.
    */
    Triangle(pair<T, T> P, pair<T, T> Q, pair<T, T> r)
        : A(P), B(Q), C(r), a(dist(A, B)), b(dist(B, C)), c(dist(C, A)) {}

    /**
    *  Time complexity: O(1)
    */
    double perimeter() { return a + b + c; }
    double inradius() { return (2 * area()) / perimeter(); }
    double circumradius() { return (a * b * c) / (4.0 * area()); }
    
    /**
    *  @return Area.
    *
    *  Time complexity: O(1)
    */
    T area() {
        T det = (A.x * B.y + A.y * C.x + B.x * C.y) -
                (C.x * B.y + C.y * A.x + B.x * A.y);
        if (is_floating_point_v<T>) return 0.5 * abs(det);
        return abs(det);
    }

    /**
    *  @return Sides class.
    *
    *  Time complexity: O(1)
    */
    Class sidesClassification() {
        if (equals(a, b) and equals(b, c)) return EQUILATERAL;
        if (equals(a, b) or equals(a, c) or equals(b, c)) return ISOSCELES;
        return SCALENE;
    }

    /**
    *  @return Angle class.
    *
    *  Time complexity: O(1)
    */
    Angles anglesClassification() {
        double alpha = acos((a * a - b * b - c * c) / (-2.0 * b * c));
        double beta  = acos((b * b - a * a - c * c) / (-2.0 * a * c));
        double gamma = acos((c * c - a * a - b * b) / (-2.0 * a * b));
        double right = acos(-1.0) / 2.0;
        if (equals(alpha, right) || equals(beta, right) || equals(gamma, right))
            return RIGHT;
        if (alpha > right || beta > right || gamma > right) return OBTUSE;
        return ACUTE;
    }

    /**
    *  @return Medians intersection point.
    *
    *  Time complexity: O(1)
    */
    pd barycenter() {
        double x = (A.x + B.x + C.x) / 3.0;
        double y = (A.y + B.y + C.y) / 3.0;
        return {x, y};
    }

    /**
    *  @return Circumcenter point.
    *
    *  Time complexity: O(1)
    */
    pd circumcenter() {
        double D = 2 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y));
        T A2 = A.x * A.x + A.y * A.y, B2 = B.x * B.x + B.y * B.y,
                                      C2 = C.x * C.x + C.y * C.y;
        double x = (A2 * (B.y - C.y) + B2 * (C.y - A.y) + C2 * (A.y - B.y)) / D;
        double y = (A2 * (C.x - B.x) + B2 * (A.x - C.x) + C2 * (B.x - A.x)) / D;
        return {x, y};
    }

    /**
    *  @return Bisectors intersection point.
    *
    *  Time complexity: O(1)
    */
    pd incenter() {
        double P = perimeter();
        double x = (a * A.x + b * B.x + c * C.x) / P;
        double y = (a * A.y + b * B.y + c * C.y) / P;
        return {x, y};
    }

    /**
    *  @return Heights intersection point.
    *
    *  Time complexity: O(1)
    */
    pd orthocenter() {
        Line<T> r(A, B), s(A, C);
        Line<T> u{r.b, -r.a, -(C.x * r.b - C.y * r.a)};
        Line<T> v{s.b, -s.a, -(B.x * s.b - B.y * s.a)};
        double det = u.a * v.b - u.b * v.a;
        double x = (-u.c * v.b + v.c * u.b) / det;
        double y = (-v.c * u.a + u.c * v.a) / det;
        return {x, y};
    }

    pair<T, T> A, B, C;
    T a, b, c;
};
```

### Polígono

```c++
template <typename T>
struct Polygon {
    /**
    *  @param  PS  Clock-wise points.
    */
    Polygon(const vector<pair<T, T>>& PS)
        : vs(PS), n(vs.size()) { vs.eb(vs.front()); }

    /**
    *  @return True if is convex.
    *
    *  Time complexity: O(N)
    */
    bool convex() {
        if (n < 3) return false;
        ll P = 0, N = 0, Z = 0;

        rep(i, 0, n) {
            auto d = D(vs[i], vs[(i + 1) % n], vs[(i + 2) % n]);
            d ? (d > 0 ? ++P : ++N) : ++Z;
        }

        return P == n or N == n;
    }

    /**
    *  @return Area. If points are integer, double the area.
    *
    *  Time complexity: O(N)
    */
    T area() {
        T a = 0;
        rep(i, 0, n) a += vs[i].x * vs[i + 1].y - vs[i + 1].x * vs[i].y;
        if (is_floating_point_v<T>) return 0.5 * abs(a);
        return abs(a);
    }
    
    /**
    *  @return Perimeter.
    *
    *  Time complexity: O(N)
    */
    double perimeter() {
        double P = 0;
        rep(i, 0, n) P += dist(vs[i], vs[i + 1]);
        return P;
    }

    /**
    *  @param  P  Point
    *  @return    True if P inside polygon.
    *
    *  Doesn't consider border points.
    *
    *  Time complexity: O(N)
    */
    bool contains(const pair<T, T>& P) {
        if (n < 3) return false;
        double sum = 0;

        rep(i, 0, n) {
            // border points are considered outside, should
            // use contains point in segment to count them
            auto d = D(vs[i], vs[i + 1], P);
            double a = angle(P, vs[i], P, vs[i + 1]);
            sum += d > 0 ? a : (d < 0 ? -a : 0);
        }

        return equals(abs(sum), 2.0 * acos(-1.0));  // check precision
    }

    /**
    *  @param  P  First point.
    *  @param  Q  Second point.
    *  @return    One of the polygons generated through the cut of the line PQ.
    *
    *  Time complexity: O(N)
    */
    Polygon cut(const pair<T, T>& P, const pair<T, T>& Q) {
        vector<pair<T, T>> points;
        double EPS { 1e-9 };

        rep(i, 0, n) {
            auto d1 = D(P, Q, vs[i]), d2 = D(P, Q, vs[i + 1]);
            if (d1 > -EPS) points.eb(vs[i]);
            if (d1 * d2 < -EPS)
                points.eb(intersection(vs[i], vs[i + 1], P, Q));
        }

        return { points };
    }

    /**
    *  @return Circumradius length.
    *
    *  Regular polygon.
    *
    *  Time complexity: O(1)
    */
    double circumradius() {
        double s = dist(vs[0], vs[1]);
        return (s / 2.0) * (1.0 / sin(acos(-1.0) / n));
    }

    /**
    *  @return Apothem length.
    *
    *  Regular polygon.
    *
    *  Time complexity: O(1)
    */
    double apothem() {
        double s = dist(vs[0], vs[1]);
        return (s / 2.0) * (1.0 / tan(acos(-1.0) / n));
    }

private:
    // lines intersection
    pair<T, T> intersection(const pair<T, T>& P, const pair<T, T>& Q,
                            const pair<T, T>& R, const pair<T, T>& S) {
        T a = S.y - R.y, b = R.x - S.x, c = S.x * R.y - R.x * S.y;
        T u = abs(a * P.x + b * P.y + c);
        T v = abs(a * Q.x + b * Q.y + c);
        return { (P.x * v + Q.x * u) / (u + v), (P.y * v + Q.y * u) / (u + v) };
    }

    vector<pair<T, T>> vs;
    ll n;
};
```

## Matemática

### Matriz

```c++
template <typename T>
struct Matrix {
    Matrix(const vector<vector<T>>& matrix) : mat(matrix), n(mat.size()) {}
    Matrix(ll m) : n(m) { mat.resize(n, vector<T>(n)); }
    vector<T>& operator[](ll i) { return mat[i]; }
    
    Matrix operator*(Matrix& other) {
        Matrix res(n);
        rep(i, 0, n) rep(j, 0, n) rep(k, 0, n)
            res[i][k] += mat[i][j] * other[j][k];
        return res;
    }
    
    /**
     *  @param  matrix  Matrix.
     *  @param  b       Exponent.
     *  @return         Matrix^b.
     *
     *  Time complexity: O(N^3 * log(B))
    */
    static Matrix pow(const Matrix& matrix, ll b) {
        ll n = matrix.n;
        Matrix tmp = matrix, res(n);
        rep(i, 0, n) res[i][i] = 1;
        while (b > 0) {
            if (b & 1) res = res * tmp;
            tmp = tmp * tmp;
            b /= 2;
        }
        return res;
    }

    vector<vector<T>> mat;
    ll n;
};
```

## Strings

### Hash

```c++
/**
 *  @brief Represent strings with integers.
*/
struct Hash {
    static const ll M1 = 1e9 + 7, M2 = 1e9 + 9, p1 = 31, p2 = 29;
    #define T pair<Mi<M1>, Mi<M2>>
    
    /**
     *  @param  s  String.
     *
     *  Time complexity: O(N)
    */
    Hash(const string& s) : n(s.size()), ps(n), pot(n) {
        T res(0, 0), b(1, 1);
        rep(i, 0, n) {
            ll v = s[i] - 'a' + 1;
            res.x *= p1, res.y *= p2;
            res.x += v, res.y += v;
            ps[i] = res;
            
            b.x *= p1, b.y *= p2;
            pot[i].x = b.x;
            pot[i].y = b.y;
        }
    }
    
    /**
     *  @param  i  First interval extreme.
     *  @param  j  Second interval extreme.
     *  @return    Pair of integers that represents the substring [i, j].
     *
     *  Time complexity: O(1)
    */
    pll get(ll i, ll j) {
        assert(i >= 0 and j >= 0 and i < n and j < n);
        T diff;
        diff.x = ps[j].x - (i ? ps[i - 1].x : 0) * pot[j - i].x;
        diff.y = ps[j].y - (i ? ps[i - 1].y : 0) * pot[j - i].y;
        return { diff.x.v, diff.y.v };
    }
    
    ll n;
    vector<T> ps, pot;
};
```

### Suffix Automaton

```c++
struct SuffixAutomaton {
    /**
    *  @param  s  String.
    *
    *  Time complexity: O(Nlog(N))
    */
    SuffixAutomaton(const string &s) {
        make_node();
        for (auto c : s) add(c);
        
        // preprocessing for count of countAndFirst
        vector<pll> order(sz - 1);
        rep(i, 1, sz) order[i - 1] = {len[i], i};
        sort(all(order), greater<>());
        for (auto [_, i] : order) cnt[lnk[i]] += cnt[i];
        
        // preprocessing for kThSub and kThDSub
        dfs(0);
    }

    /**
    *  @param  t  String.
    *  @return    Pair with how many times substring t
    *             appears and index of first occurrence.
    *
    *  Time complexity: O(M)
    */
    pll countAndFirst(const string &t) {
        ll u = 0;
        for (auto c : t) {
            ll v = next[u][c - 'a'];
            if (!v) return {0, -1};
            u = v;
        }
        return {cnt[u], fpos[u] - t.size() + 1};
    }

    /**
    *  @returns  Quantity of distinct substrings. 
    *
    *  Time complexity: O(N)
    */
    ll dSubs() {
        ll res = 0;
        rep(i, 1, sz)
            res += len[i] - len[lnk[i]];
        return res;
    }

    /**
    *  @returns  Vector with quantity of distinct substrings of each size. 
    *
    *  Time complexity: O(N)
    */
    vll dSubsBySz() {
        vll hs(sz, -1);
        hs[0] = 0;
        queue<ll> q;
        q.emplace(0);
        ll mx = 0;
        while (!q.empty()) {
            ll u = q.front();
            q.pop();
            rep(i, 0, alpha) {
                ll v = next[u][i];
                if (!v) continue;
                if (hs[v] == -1) {
                    q.emplace(v);
                    hs[v] = hs[u] + 1;
                    mx    = max(mx, len[v]);
                }
            }
        }

        vll res(mx);
        rep(i, 1, sz) {
            ++res[hs[i] - 1];
            if (len[i] < mx) --res[len[i]];
        }
        rep(i, 1, mx) res[i] += res[i - 1];
        return res;
    }

    /**
    *  @param  k  Value, starts from 0.
    *  @return    k-th substring lexographically. 
    *
    *  Time complexity: O(N)
    */
    string kThSub(ll k) {
        k += n;
        string res;
        ll u = 0;
        while (k >= cnt[u]) {
            k -= cnt[u];
            rep(i, 0, alpha) {
                ll v = next[u][i];
                if (!v) continue;
                if (rcnt[v] > k) {
                    res += i + 'a', u = v;
                    break;
                }
                k -= rcnt[v];
            }
        }
        return res;
    }

    /**
    *  @param  k  Value, starts from 0.
    *  @return    k-th distinct substring lexographically. 
    *
    *  Time complexity: O(N)
    */
    string kThDSub(ll k) {
        string res;
        ll u = 0;
        while (k >= 0) {
            rep(i, 0, alpha) {
                ll v = next[u][i];
                if (!v) continue;
                if (dcnt[v] > k) {
                    res += i + 'a', --k, u = v;
                    break;
                }
                k -= dcnt[v];
            }
        }
        return res;
    }

private:
    ll make_node(ll _len = 0, ll _fpos = -1, ll _lnk = -1, ll _cnt = 0,
                 ll _rcnt = 0, ll _dcnt = 0) {
        next.eb(vll(alpha));
        len.eb(_len), fpos.eb(_fpos), lnk.eb(_lnk);
        cnt.eb(_cnt), rcnt.eb(_rcnt), dcnt.eb(_dcnt);
        return sz++;
    }

    void add(char c) {
        c -= 'a', ++n;
        ll u = make_node(len[last] + 1, len[last], 0, 1);
        ll p = last;
        while (p != -1 and !next[p][c]) {
            next[p][c] = u;
            p = lnk[p];
        }
        if (p == -1) lnk[u] = 0;
        else {
            ll q = next[p][c];
            if (len[p] + 1 == len[q]) lnk[u] = q;
            else {
                ll v = make_node(len[p] + 1, fpos[q], lnk[q]);
                next[v] = next[q];
                while (p != -1 and next[p][c] == q) {
                    next[p][c] = v;
                    p = lnk[p];
                }
                lnk[q] = lnk[u] = v;
            }
        }
        last = u;
    }

    void dfs(ll u) {
        dcnt[u] = 1, rcnt[u] = cnt[u];
        rep(i, 0, alpha) {
            ll v = next[u][i];
            if (!v) continue;
            if (!dcnt[v]) dfs(v);
            dcnt[u] += dcnt[v];
            rcnt[u] += rcnt[v];
        }
    }

    vvll next;
    vll len, fpos, lnk, cnt, rcnt, dcnt;
    ll sz = 0, last = 0, n = 0, alpha = 26;
};
```

## Outros

### Soma de prefixo 2D

```c++
/**
 *  @brief Make rectangular interval sum queries.
*/
template <typename T>
struct Psum2D {
    /**
     *  @param  xs  Matrix.
     *
     *  Time complexity: O(N^2)
    */
    Psum2D(const vector<vector<T>>& xs)
        : n(xs.size()), m(xs[0].size()), psum(n + 1, vector<T>(m + 1)) {
        rep(i, 0, n)
            rep(j, 0, m) {
                // sum side and up rectangles, add element and remove intersection
                psum[i + 1][j + 1] = psum[i + 1][j] + psum[i][j + 1];
                psum[i + 1][j + 1] += xs[i][j] - psum[i][j];
            }
    }

    /**
     *  @param  ly  Lower y bound.
     *  @param  lx  Lower x bound.
     *  @param  hy  Higher y bound.
     *  @param  hx  Higher x bound.
     *  @return     Sum in that rectangle.
     *
     *  Time complexity: O(1)
    */
    T query(ll ly, ll lx, ll hy, ll hx) {
        // sum total rectangle, subtract side and up and add intersection
        T res = psum[hy][hx] - psum[hy][lx - 1] - psum[ly - 1][hx];
        res += psum[ly - 1][lx - 1];
        return res;
    }

    ll n, m;
    vector<vector<T>> psum;
};
```

### Soma de prefixo 3D

```c++
/**
 *  @brief Make cuboid interval sum queries.
*/
template <typename T>
struct Psum3D {
    /**
     *  @param  xs  3D Matrix.
     *
     *  Time complexity: O(N^3)
    */
    Psum3D(const vector<vector<vector<T>>>& xs)
            : n(xs.size()), m(xs[0].size()), o(xs[0][0].size()),
              psum(n + 1, vector<vector<T>>(m + 1, vector<T>(o + 1)) {
        rep(i, 1, n + 1) rep(j, 1, m + 1) rep(k, 1, o + 1) {
            // sum cuboids from sides and down
            psum[i][j][k] = psum[i - 1][j][k] + psum[i][j - 1][k] +
                                                psum[i][j][k - 1];
            // subtract intersections
            psum[i][j][k] -= psum[i][j - 1][k - 1] + psum[i - 1][j][k - 1] +
                                                     psum[i - 1][j - 1][k];
            // re-sum missing cuboid and add element
            psum[i][j][k] += psum[i - 1][j - 1][k - 1] + xs[i - 1][j - 1][k - 1];
        }
    }

    /**
     *  @param  ly  Lower y bound.
     *  @param  lx  Lower x bound.
     *  @param  lz  Lower z bound.
     *  @param  hy  Higher y bound.
     *  @param  hx  Higher x bound.
     *  @param  hz  Higher z bound.
     *  @return     Sum in that cuboid.
     *
     *  Time complexity: O(1)
    */
    T query(ll lx, ll ly, ll lz, ll hx, ll hy, ll hz) {
        // sum total cuboid, subtract sides and down
        T res = psum[hx][hy][hz]     - psum[lx - 1][hy][hz] -
                 psum[hx][ly - 1][hz] - psum[hx][hy][lz - 1];
        // add intersections
        res += psum[hx][ly - 1][lz - 1] + psum[lx - 1][hy][lz - 1] +
                                          psum[lx - 1][ly - 1][hz];
        // re-subtract missing cuboid
        res -= psum[lx - 1][ly - 1][lz - 1];
        return res;
    }

    ll n, m, o;
    vector<vector<vector<T>>> psum;
};
```

# Utils

### Aritmética modular

```c++
const ll MOD = 1e9 + 7;

/**
 *  @brief Modular arithmetics.
*/
template <ll M = MOD>
struct Mi {
    ll v;   
    Mi() : v(0) {}
    Mi(ll x) : v(x % M) { v += (v < 0) * M; }
    friend bool operator==(Mi a, Mi b) { return a.v == b.v; }
    friend bool operator!=(Mi a, Mi b) { return a.v != b.v; }
    friend ostream& operator<<(ostream& os, Mi a) { return os << a.v; }
    Mi operator+=(Mi b) { return v += b.v - (v + b.v >= M) * M; }
    Mi operator-=(Mi b) { return v -= b.v - (v - b.v  < 0) * M; }
    Mi operator*=(Mi b) { return v = v * b.v % M; }
    Mi operator/=(Mi b) & { return *this *= pow(b, M - 2); }
    friend Mi operator+(Mi a, Mi b) { return a += b; }
    friend Mi operator-(Mi a, Mi b) { return a -= b; }
    friend Mi operator*(Mi a, Mi b) { return a *= b; }
    friend Mi operator/(Mi a, Mi b) { return a /= b; }
    static Mi pow(Mi a, ll b) {
        Mi res = 1;
        while (b) {
            if (b & 1) res *= a;
            a *= a;
            b /= 2;
        }
        return res;
    }
};
```

### Big integer

```c++
/**
 *  @brief Integers bigger than long long using string.
*/
struct Bi {
    Bi() : v("0") {}
    Bi(const string& x) : v(x) { reverse(all(v)); }
    friend Bi operator+(Bi a, const Bi& b) { return a += b; }
    friend Bi operator-(Bi a, const Bi& b) { return a -= b; }

    friend ostream& operator<<(ostream& os, const Bi& a) {
        ll i = a.v.size() - 1;
        while (a.v[i] == '0' and i > 0) --i;
        while (i >= 0) os << a.v[i--];
        return os;
    }

    // Time complexity: O(N)
    Bi operator+=(const Bi& b) {
        bool c = false;
        rep(i, 0, max(v.size(), b.v.size())) {
            ll x = c;
            if (i < v.size()) x += v[i] - '0';
            if (i < b.v.size()) x += b.v[i] - '0';
            c = x >= 10, x -= 10 * (x >= 10);
            if (i < v.size()) v[i] = x + '0';
            else v += x + '0';
        }
        if (c) v += '1';
        return *this;
    }

    /**
    * assumes a > b
    *
    * Time complexity: O(N)
    */
    Bi operator-=(const Bi& b) {
        rep(i, 0, v.size()) {
            ll x = v[i] - '0';
            if (i < b.v.size()) x -= b.v[i] - '0';
            if (x < 0) x += 10, --v[i + 1];
            v[i] = x + '0';
        }
        return *this;
    }

    /**
     *  @param  n  Size of prefix.
     *  @return    Prefix.
     *
     *  Time complexity: O(N)
    */
    Bi prefix(ll n) {
        string p = v.substr(v.size() - n, n);
        reverse(all(p));
        return p;
    }

    /**
     *  @param  n  Size of suffix.
     *  @return    Suffix.
     *
     *  Same as x % 10^(n-1)
     *
     *  Time complexity: O(N)
    */
    Bi suffix(ll n) {
        string s = v.substr(0, n);
        reverse(all(s));
        return s;
    }
    
    string v;
};
```

### Ceil division

```c++
/**
 *  @param  a  Numerator.
 *  @param  b  Denominator.
 *  @return    Result of ceil division.
 *
 *  Time complexity: O(1)
*/
ll ceilDiv(ll a, ll b) { assert(b != 0); return a / b + ((a ^ b) > 0 && a % b != 0); }
```

### Conversão de índices

```c++
#define K(i, j) ((i) * w + (j))
#define I(k)    ((k) / w)
#define J(k)    ((k) % w)
```

### Compressão de coordenadas

```c++
/**
 *  @brief      Compress values from a vector.
 *  @param  xs  Vector.
 *  @return     Maps with the compressed values and uncompressed values.
 *
 *  Time complexity: O(Nlog(N))
*/
template <typename T>
pair<map<T, ll>, map<ll, T>> compress(vector<T>& xs) {
    ll i = 0;
    set<T> ys(all(xs));
    map<ll, T> pm;
    map<T, ll> mp;
    for (T y : ys) {
        pm[i] = y;
        mp[y] = i++;
    }
    return mp;
}
```

### Fatos

Bitwise

> `a + b = (a & b) + (a | b)`

> `a + b = a ^ b + 2 * (a & b)`

Geometria

> Sendo `A` a área da treliça, `I` a quantidade de pontos interiores
  com coordenadas inteiras e `B` os pontos da borda com coordenadas
  inteiras, `A = I + B / 2 - 1`. Assim como, `I = (2A + 2 - B) / 2`

> Sendo `y/x` o coeficiente angular de uma reta com coordenadas
  inteiras, `gcd(y, x)` representa a quantidade de pontos inteiros nela

> Ao trabalhar com distância de Manhattam podemos fazer a transformação
  `(x, y) -> (x + y, x - y)` para transformar os pontos e ter uma equivalência
  com a distância de Chebyshev, de forma que agora conseguimos tratar `x` e `y`
  separadamente, fazer boundig boxes, etc

Matemática

> A quantidade de divisores de um número é a multiplicação de cada potência
  da fatoração `+ 1`

> Maior quantidade de divisores de um número `< 10^18` é `107520`

> Maior quantidade de divisores de um número `< 10^6` é `240`

> Maior quantidade de divisores de um número `< 10^3` é `32`

> Maior diferença entre dois primos consecutivos `< 10^18` é `1476`
  (Podemos concluir também que a partir de um número arbitrário a 
   distância para um coprimo é bem menor que esse valor)

> Maior quantidade de primos na fatoração de um número `< 10^6` é `19`

> Maior quantidade de primos na fatoração de um número `< 10^3` é `9`

> Números primos interessantes: `2^31 - 1, 2^31 + 11, 10^18 - 11, 10^18 + 3`.

> `gcd(a, b) = gcd(a, a - b)`, `gcd(a, b, c) = gcd(a, a - b, a - c)`

> Divisibilidade por `3`: Soma dos algarismos divisível por `3`

> Divisibilidade por `4`: Número formado pelos dois últimos algarismos divísivel por `4`

> Divisibilidade por `6`: Par e divísivel por `3`

> Divisibilidade por `7`: Dobro do último algarismo subtraído do número sem ele divisível por `7` (pode ir repetindo)
ou a soma alternada de blocos de três algarismos divisível por `7`

> Divisibilidade por `8`: Número formado pelos três últimos algarismos divísivel por `8`

> Divisibilidade por `9`: Soma dos algarismos divisível por `9`

> Divisibilidade por `11`: Soma alternada dos algarismos divisível por `11`

> Divisibilidade por `12`: Se for divisível por `3` e `4`

Strings

> Sejam `p` e `q` dois períodos de uma string `s`. Se `p + q − mdc(p, q) ≤ |s|`,
  então `mdc(p, q)` também é período de `s`

> Relação entre bordas e períodos: A sequência `|s| − |border(s)|, |s| − |border^2(s)|, ..., |s| − |border^k(s)|`
  é a sequência crescente de todos os possíveis períodos de `s`

Outros

> Princípio da inclusão e exclusão: a união de `n` conjuntos é
  a soma de todas as interseções de um número ímpar de conjuntos menos
  a soma de todas as interseções de um número par de conjuntos
  
> A regra de Warnsdorf é uma heurística para encontrar um caminho em
  que o cavalo passa por todas as casas uma única vez: sempre escolher
  o próximo movimento para a casa com o menor número de casas alcançáveis.
  Talvez funcione em outros cenários.

### Igualdade flutuante

```c++
/**
 *  @param  a  First value.
 *  @param  b  Second value.
 *  @return    True if they are equal.
 *
 *  Time complexity: O(1)
*/
template <typename T, typename S>
bool equals(T a, S b) { return abs(a - b) < 1e-9; }
```

### Intervalos com soma S

```c++
/**
 *  @param  xs   Vector.
 *  @param  sum  Desired sum.
 *  @return      Quantity of contiguous intervals with sum S.
 *
 *  Can change to count odd/even sum intervals (hist of even and odd).
 *  Also could change to get contiguos intervals with sum bigger equal,
 *  using an ordered-set.
 *
 *  Time complexity: O(Nlog(N))
*/
template <typename T>
ll countIntervals(const vector<T>& xs, ll sum) {
    map<T, ll> hist;
    hist[0] = 1;
    ll ans = 0;
    T csum = 0;
    for (T x : xs) {
        csum += x;
        ans += hist[csum - sum];
        ++hist[csum];
    }
    return ans;
}
```

### Kadane

```c++
/**
 *  @param  xs  Vector.
 *  @param  mx  Maximum Flag (true if want max).
 *  @return     Max/min contiguous sum and smallest interval inclusive.
 *
 *  We consider valid an empty sum.
 *
 *  Time complexity: O(N)
*/
template <typename T>
tuple<T, ll, ll> kadane(const vector<T>& xs, bool mx = true) {
    T res = 0, csum = 0, l = -1, r = -1, j = 0;
    rep(i, 0, xs.size()) {
        csum += xs[i] * (mx ? 1 : -1);
        if (csum < 0) csum = 0, j = i + 1;  //            > if wants biggest interval
        else if (csum > res or (csum == res and i - j + 1 < r - l + 1))
            res = csum, l = j, r = i;
    }
    return { res * (mx ? 1 : -1), l, r };
}
```

### Overflow check

```c++
// BEGIN OVERFLOW CHECK --------------------------------|
ll mult(ll a, ll b) {
    if (abs(a) >= LLONG_MAX / abs(b))
        return LLONG_MAX;  // overflow
    return a * b;
}

ll sum(ll a, ll b) {
    if (abs(a) >= LLONG_MAX - abs(b))
        return LLONG_MAX;  // overflow
    return a + b;
}
// END OVERFLOW CHECK ----------------------------------|
```

### Pares com gcd x

```c++
/**
 *  @param  xs  Target.
 *  @param  x   Desired gcd.
 *  @return     Quantity of pairs with gcd equals x.
 *
 *  Time complexity: O(Nlog(N))/O(1)
*/
vll gcdPairs(const vll& xs, ll x) {
    const ll MAXN = 1e6 + 1;
    static vll dp(MAXN, -1), ms(MAXN), hist(MAXN);
    if (dp[1] == -1) {
        for (ll x : xs)
            ++hist[x];
        
        rep(i, 1, MAXN)
            for (ll j = i; j < MAXN; j += i)
                ms[i] += hist[j];
        
        per(i, MAXN - 1, 1) {
            dp[i] = ms[i] * (ms[i] - 1) / 2;
            for (ll j = 2 * i; j < MAXN; j += i)
                dp[i] -= dp[j];
        }
    }
    return dp[x];
}
```

### Próximo maior/menor elemento

```c++
/**
 *  @param  xs  Vector.
 *  @return     Vector of indexes of closest biggest.
 *
 *  Example: c[i] = j where j < i and xs[j] > xs[i] and it's the closest.
 *  If there isn't then c[i] = -1.
 *
 *  Time complexity: O(N)
*/
template <typename T>
vector<T> closests(const vector<T>& xs) {
    vll c(xs.size(), -1);  // n if to the right
    stack<ll> prevs;
    // n - 1 -> 0: to the right
    rep(i, 0, xs.size()) {  //                    <= if want smallest
        while (!prevs.empty() and xs[prevs.top()] <= xs[i])
            prevs.pop();
        if (!prevs.empty()) c[i] = prevs.top();
        prevs.emplace(i);
    }
    return c;
}
```

### Soma de todos os intervalos

```c++
/**
 *  @param  xs  Vector.
 *  @return     Sum of all intervals.
 *
 *  By counting in how many intervals the element appear
 *
 *  Time complexity: O(N)
*/
template <typename T>
T sumAllIntervals(const vector<T>& xs) {
    T sum = 0;
    ll opens = 0;
    rep(i, 0, xs.size()) {
        opens += xs.size() - 2 * i;
        sum += xs[i] * opens;
    }
    return sum;
}
```

```c++
/**
 *  @param  xs  Vector.
 *  @return     Sum of all intervals.
 *
 *  By adding each prefix sum
 *
 *  Time complexity: O(N)
*/
T sumAllIntervals(const vector<T>& xs) {
    ll n = xs.size();
    T sum = 0, csum = 0;
    rep(i, 0, n)
        csum += xs[i] * (n - i);
    rep(i, 0, n) {
        sum += csum;
        csum -= xs[i] * (n - i);
    }
    return sum;
}
```
