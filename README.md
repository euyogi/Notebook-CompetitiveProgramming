# Template

```c++
#include <bits/stdc++.h>

using namespace std;
using ll = long long;
using ull = unsigned long long;
using pll = pair<long long, long long>;
using vll = vector<long long>;
using vvll = vector<vector<long long>>;
using vpll = vector<pair<long long, long long>>;
using Point = pair<long long, long long>;

constexpr ll oo = numeric_limits<ll>::max();

void solve() {
    return;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int testes = 1;
    // cin >> testes;

    while (testes--)
        solve();
}
```

# Igualdade entre pontos flutuantes

```c++
template<typename T>
bool equals(T a, T b) {
    constexpr double EPS { 1e-9 };

    return std::is_floating_point<T>::value ?  fabs(a - b) < EPS : a == b;
}
```
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
Point rotatePoint(const Point& P, double angleRadians) {
    #define x first
    #define y second

    double x = P.x * cos(angleRadians) - P.y * sin(angleRadians);
    double y = P.x * sin(angleRadians) + P.y * cos(angleRadians);

    return {x, y};
}
```

Checar orientação do ponto P em relação à reta AB

```c++
// D = 0: P pertence a reta
// D > 0: P à esquerda da reta
// D < 0: P à direita da reta
ll D(const Point& A, const Point& B, const Point& P) {
    #define x first
    #define y second

    return (A.x * B.y + A.y * P.x + B.x * P.y) - (P.x * B.y + P.y * A.x + B.x * A.y);
}
```

Checar ângulo entre as retas AB e CD

```c++
double angle(const Point& A, const Point& B, const Point& C, const Point& D) {
    #define x first
    #define y second

    auto ux = A.x - B.x;
    auto uy = A.y - B.y;

    auto vx = C.x - D.x;
    auto vy = C.y - D.y;

    auto num = ux * vx + uy * vy;
    auto den = hypot(ux, uy) * hypot(vx, vy);

    // Caso especial: se den == 0, algum dos vetores é degenerado: os dois
    // pontos são iguais. Neste caso, o ângulo não está definido

    return acos(num / den);
}
```

# Linha

```c++
template<typename T>
class Line {
public:
    T a, b, c;

    #define x first
    #define y second

    Line(const Point& A, const Point& B)
        : a(A.y - B.y), b(B.x - A.x), c(A.x * B.y - B.x * A.y) {
        if (a < 0 or (a == 0 and b < 0)) {
            a *= -1;
            b *= -1;
            c *= -1;
        }

        if (std::is_floating_point<T>::value) {
            auto tmp = a;
            a /= tmp;
            b /= tmp;
            c /= tmp;
        }
        else { 
            ll gcd_abc = gcd(a, gcd(b, c));
            a /= gcd_abc;
            b /= gcd_abc;
            c /= gcd_abc;
        }
    }

    bool operator==(const Line& r) const {
        auto k = a ? a : b;
        auto s = r.a ? r.a : r.b;

        return equals(a*s, r.a*k) && equals(b*s, r.b*k) && equals(c*s, r.c*k);
    }

    bool parallel(const Line& r) const {
        auto det = a*r.b - b*r.a;
        return det == 0 and !(*this == r);
    }

    bool orthogonal(const Line& r) const { return equals(a * r.a + b * r.b, 0); }

    // Distância da reta pro ponto P
    double distance(const Point& P) const { return fabs(a*P.x + b*P.y + c)/hypot(a, b); }

    Point closest(const Point& P) const { // Ponto da reta mais próximo de P
        auto den = a*a + b*b;

        auto x = (b*(b*P.x - a*P.y) - a*c)/den;
        auto y = (a*(-b*P.x + a*P.y) - b*c)/den;

        return { x, y };
    }

private:
    bool equals(T a, T b) const {
        constexpr double EPS { 1e-9 };

        return std::is_floating_point<T>::value ?  fabs(a - b) < EPS : a == b;
    }
};
```

# Segmento

```c++
template<typename T>
class Segment {
public:
    Point A, B;

    #define x first
    #define y second

    Segment(const Point& A, const Point& B) : A(A), B(B) {}

    bool contains(const Point& P) const {
        auto xmin = min(A.x, B.x);
        auto xmax = max(A.x, B.x);
        auto ymin = min(A.y, B.y);
        auto ymax = max(A.y, B.y);

        if (P.x < xmin || P.x > xmax || P.y < ymin || P.y > ymax)
            return false;

        return equals((P.y - A.y)*(B.x - A.x), (P.x - A.x)*(B.y - A.y));
    }

    bool intersect(const Segment& s) const {
        auto d1 = D(A, B, s.A);
        auto d2 = D(A, B, s.B);

        if ((equals(d1, 0) && contains(s.A)) || (equals(d2, 0) && contains(s.B)))
            return true;

        auto d3 = D(s.A, s.B, A);
        auto d4 = D(s.A, s.B, B);

        if ((equals(d3, 0) && s.contains(A)) || (equals(d4, 0) && s.contains(B)))
            return true;

        return (d1 * d2 < 0) && (d3 * d4 < 0);
    }

private:
    bool equals(T a, T b) const {
        constexpr double EPS { 1e-9 };

        return std::is_floating_point<T>::value ?  fabs(a - b) < EPS : a == b;
    }

    ll D(const Point& A, const Point& B, const Point& P) const {
        return (A.x * B.y + A.y * P.x + B.x * P.y) - (P.x * B.y + P.y * A.x + B.x * A.y);
    }
};
```
