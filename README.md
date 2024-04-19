# BIT Tree

Somar valores em intervalos.

```c++
class BITree {
public:
    BITree(int n) : ts(n + 2, 0), N(n) {}

    ll value_at(int i) { return RSQ(i); }

    void range_add(int i, int j, ll x) {
        add(i, x);
        add(j + 1, -x);
    }

private:
    vector<ll> ts;
    int N;

    int LSB(int n) { return n & (-n); }

    ll RSQ(int i) {
        ll sum = 0;

        while (i >= 1) {
            sum += ts[i];
            i -= LSB(i);
        }

        return sum;
    }

    void add(int i, ll x) {
        while (i <= N) {
            ts[i] += x;
            i += LSB(i);
        }
    }
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
using pll = pair<ll, ll>;
using vpll = vector<pll>;
using vll = vector<ll>;

constexpr ll oo = numeric_limits<ll>::max();

vll djikstra(vector<vpll>& g, int s) {
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

# Rotacionar Ponto

```c++
Point rotatePoint(const Point& p, double angleRadians) {
    double x = p.first * cos(angleRadians) - p.second * sin(angleRadians);
    double y = p.first * sin(angleRadians) + p.second * cos(angleRadians);

    return {x, y};
}
```

# Reta AB contém ponto P

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
