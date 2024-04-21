# BIT Tree

Somar valores em intervalos.

```c++
class BITree {
public:
    BITree(size_t n) : m_ts(n + 1, 0) {}

    ll valueAt(size_t i) { return RSQ(i); }

    void rangeAdd(size_t i, size_t j, ll x) {
        add(i, x);
        add(j + 1, -x);
    }

private:
    ll LSB(ll n) { return n & (-n); }

    ll RSQ(ll i) {
        ll sum = 0;

        while (i >= 1) {
            sum += m_ts[i];
            i -= LSB(i);
        }

        return sum;
    }

    void add(size_t i, ll x) {
        while (i < m_ts.size()) {
            m_ts[i] += x;
            i += LSB(i);
        }
    }

    vector<ll> m_ts;
};
```

# Red-Black Tree

Armazenar valores ordenadamente e conseguir acessar
o índice do primeiro elemento maior ou igual a x

```c++
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace __gnu_pbds;

typedef tree<ll, null_type, less_equal<ll>,
rb_tree_tag,tree_order_statistics_node_update> set_t;

int main() {
    set_t dist;
    dist.order_of_key(...);
    dist.insert(...);
}
```

# Djikstra

Medir a menor distância de cada aresta para uma principal

```c++
using pll = pair<long long, long long>;
using vpll = vector<pair<long long, long long>>;
using vll = vector<long long>;

constexpr ll oo = numeric_limits<ll>::max();

vll djikstra(const vector<vpll>& g, ll s) {
    vll dists(g.size(), oo);
    priority_queue<pll, vpll, greater<>> pq;
    
    pq.emplace(s, 0);
    dists[s] = 0;
    
    while (!pq.empty()) {
        auto [c, t] = pq.top();
        pq.pop();
    
        for (auto [n, w] : g[c])
            if (t + w < dists[n]) {
                pq.emplace(n, t + w);
                dists[n] = t + w;
            }
    }
    
    return dists;
}
```

# Disjoint Set Union

Representar conjuntos de elementos, conseguir saber de qual conjunto
um elemento é conseguir saber quantos elementos existem nesse conjunto

```c++
class DSU {
public:
    DSU(size_t n) : m_parent(n), m_size(n, 1) {
        iota(m_parent.begin(), m_parent.end(), 0);
    }

    ll setOf(ull x) {
        return m_parent[x] == x ? x : m_parent[x] = setOf(m_parent[x]);
    }

    bool sameSet(ull x, ull y) { return setOf(x) == setOf(y); }

    void mergeSetsOf(ull x, ull y) {
        ull a = setOf(x);
        ull b = setOf(y);

        if (a == b) return;
        if (m_size[a] > m_size[b]) swap(a, b);

        m_parent[a] = b;
        m_size[b] += m_size[a];
        m_size[a] = 0;
    }

    size_t size() { return m_parent.size(); }
    size_t sizeOfSet(ll i) { return m_size[i]; }

private:
    vector<ull> m_parent, m_size;
};
```

# Geometria

Rotacionar Ponto

```c++
Point rotatePoint(const Point& p, double angleRadians) {
    double x = p.first * cos(angleRadians) - p.second * sin(angleRadians);
    double y = p.first * sin(angleRadians) + p.second * cos(angleRadians);

    return {x, y};
}
```

Checar se a reta AB contém o ponto P

```c++
bool contains(const Point& A, const Point& B, const Point& P) {
    auto xmin = min(A.first, B.first);
    auto xmax = max(A.first, B.first);
    auto ymin = min(A.second, B.second);
    auto ymax = max(A.second, B.second);

    if (P.first < xmin || P.first > xmax || P.second < ymin || P.second > ymax)
        return false;

    return (P.second - A.second)*(B.first - A.first) == (P.first - A.first)*(B.second - A.second);
}
```
