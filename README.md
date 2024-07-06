# Notebook

## Sumário

* Template
* Utils
  * Direções adjascentes
  * Igualdade flutuante
  * Fatos
* Estruturas
  * Reta
  * Segmento
  * Círculo *
  * BIT Tree
  * Red-Black Tree
  * Segment Tree *
  * Disjoint Set Union
* Algoritmos
  * Kruskal (Árvore Geradora Mínima)
  * Busca binária
  * DFS
  * BFS *
  * Dijkstra
  * Binary Lifting
  * Menor Ancestral Comum *
  * Divisores
  * Fatoração
  * Crivo de Eratóstenes
  * Rotação de Ponto
  * Orientação de Ponto

## Main template

```c++
#include <bits/stdc++.h>

#ifdef DEBUG
#include "dbg.h"
#else
#define dbg(...)
#endif

using namespace std;
using ll = long long;
using vll = vector<ll>;
using vvll = vector<vll>;
using pll = pair<ll, ll>;
using vpll = vector<pll>;
using vvpll = vector<vpll>;
#define all(xs) xs.begin(), xs.end()
#define found(x, xs) (xs.find(x) != xs.end())

void solve() {
    
}

signed main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    ll _ts = 1;
    // cin >> _ts; cin.ignore();
    while (_ts--) { solve(); }
}
```

## Utils

### 4 direções adjascentes:

```c++
vpll ds { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
```

### 8 direções adjascentes:

```c++
vpll ds { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }, { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 } };
```

### Loop das direções:

```c++
for (auto [ox, oy] : ds) {
    ll nx = x + ox, ny = y + oy;
    // processing
}
```

### Igualdade flutuante:

```c++
template<typename T>
bool equals(T a, T b) {
    constexpr double EPS { 1e-9 };
    return std::is_floating_point<T>::value ?  fabs(a - b) < EPS : a == b;
}
```

### Fato: `a + b = (a & b) + (a | b)`

## Estruturas

### Reta

Parâmetros:

* `A` e `B`: pontos pertencentes à reta

Métodos:

* `parallel(r)`: Retorna verdadeiro se as retas são paralelas, falso caso contrário
* `orthogonal(r)`: Retorna verdadeiro se as retas são perpendiculares, falso caso contrário
* `distance(P)`: Retorna a distância entre a reta e o ponto `P`
* `closest(P)`: Retorna o ponto na reta mais do ponto `P`

```c++
class Line {
public:
    #define x first
    #define y second
    ll a, b, c;

    Line(const pll& A, const pll& B)
        : a(A.y - B.y), b(B.x - A.x), c(A.x * B.y - B.x * A.y) {
        if (a < 0 or (a == 0 and b < 0)) {
            a *= -1; b *= -1; c *= -1;
        }
        ll gcd_abc = gcd(a, gcd(b, c));
        a /= gcd_abc; b /= gcd_abc; c /= gcd_abc;
    }

    bool operator==(const Line& r) const {
        auto k = a ? a : b;
        auto s = r.a ? r.a : r.b;
        return (a * s == r.a * k) && (b * s == r.b * k) && (c * s == r.c * k);
    }

    bool parallel(const Line& r) const {
        auto det = a * r.b - b * r.a;
        return det == 0 and !(*this == r);
    }

    bool orthogonal(const Line& r) const { return (a * r.a + b * r.b == 0); }

    // distance from line to P
    double distance(const Point& P) const { return fabs(a * P.x + b * P.y + c)/hypot(a, b); }

    pll closest(const pll& P) const { // closest line point to P
        auto den = a * a + b * b;
        auto x = (b *  (b * P.x - a * P.y) - a * c)/den;
        auto y = (a * (-b * P.x + a * P.y) - b * c)/den;
        return { x, y };
    }
};
```

### Segmento

Parâmetros:

* `A` e `B`: pontos extremos do segmento

Métodos:

* `contains(P)`: Retorna verdadeiro se a reta contém o ponto `P`, falso caso contrário
* `intersect(r)`: Retorna verdadeiro se os segmentos se intersectam, falso caso contrário

```c++
class Segment {
public:
    #define x first
    #define y second
    pll A, B;

    Segment(const pll& A, const pll& B) : A(A), B(B) {}

    bool contains(const pll& P) const {
        auto xmin = min(A.x, B.x);
        auto xmax = max(A.x, B.x);
        auto ymin = min(A.y, B.y);
        auto ymax = max(A.y, B.y);
        if (P.x < xmin || P.x > xmax || P.y < ymin || P.y > ymax)
            return false;
        return (P.y - A.y)*(B.x - A.x) == (P.x - A.x)*(B.y - A.y);
    }

    bool intersect(const Segment& r) const {
        auto d1 = D(A, B, r.A);
        auto d2 = D(A, B, r.B);
        if (((d1 == 0) && contains(r.A)) || ((d2 == 0) && contains(r.B)))
            return true;
        auto d3 = D(r.A, r.B, A);
        auto d4 = D(r.A, r.B, B);
        if (((d3 == 0) && r.contains(A)) || ((d4 == 0) && r.contains(B)))
            return true;
        return (d1 * d2 < 0) && (d3 * d4 < 0);
    }

private:
    ll D(const pll& A, const pll& B, const pll& P) const {
        return (A.x * B.y + A.y * P.x + B.x * P.y) - (P.x * B.y + P.y * A.x + B.x * A.y);
    }
};
```

### BIT Tree

Parâmetros:

* `n`: intervalo máximo para as operações `[1, n]`

Métodos:

* `rangeAdd(i, j, x)`: soma `x` em cada elemento no intervalo `[i, j]`
* `rangeQuery(i, j)`: retorna a soma do intervalo `[i, j]`

```c++
class BIT {
public:
    BIT(size_t n) : bt1(n+1), bt2(n+1) {}

    void rangeAdd(size_t i, size_t j, ll x) {
    	add(i, x, bt1);           add(j + 1, -x, bt1);
    	add(i, x * (i - 1), bt2); add(j + 1, -x * j, bt2);
    }
    
    ll rangeQuery(size_t i, size_t j) {
    	return rsq(j, bt1) * j           - rsq(j, bt2) -
    	      (rsq(i - 1, bt1) * (i - 1) - rsq(i - 1, bt2));
    }

private:
    void add(size_t i, ll x, vll& bt) {
        for (; i < bt.size(); i += i & -i)
            bt[i] += x;
    }
    
    ll rsq(size_t i, vll& bt) {
        ll sum = 0;
        for (; i >= 1 i -= i & -i)
            sum += bt[i];
        return sum;
    }

    vll bt1, bt2;
};
```

### BIT Tree (Simples)

Parâmetros:

* `n`: intervalo máximo para as operações `[1, n]`

Métodos:

* `add(i, x)`: soma `x` em cada elemento no intervalo `[1, i]`
* `rsq(i)`: retorna a soma do intervalo `[1, i]`

```c++
struct BIT {
    vll bt
    BIT(size_t n) : bt(n+1) {}
    
    void add(ll i, ll x) {
        for (; i < bt.size(); i += i & -i)
            bt[i] += x;
    }
    
    ll rsq(ll i) {
        ll sum = 0;
        for (; i >= 1; i -= i & -i)
            sum += bt[i];
        return sum;
    }
};
```

### Red-Black Tree

Métodos:

* `insert(x)`: insere elemento `x`
* `order_of_key(x)`: retorna quantos elementos existem menor que `x`

```c++
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace __gnu_pbds;

// if less<>, then unique elements
typedef tree<ll, null_type, less_equal<>,
rb_tree_tag, tree_order_statistics_node_update> RBT;
```

### Disjoint Set Union

Parâmetros:

* `n`: intervalo máximo para operações `[1, n]`

Métodos:

* `mergeSetsOf(x, y)`: combina os conjuntos que contém `x` e `y`
* `sameSet(x, y)`: retorna verdadeiro se `x` e `y` estão contidos no mesmo conjunto, falso caso contrário
* `setOf(x)`: retorna o representante do conjunto que contém `x`
* `sizeOfSet(s)`: retorna quantos elementos estão contidos no conjunto representado por `s`

```c++
class DSU {
public:
    DSU(size_t n) : parent(n + 1), size(n + 1, 1) {
        iota(all(parent), 0);
    }

    ll setOf(ull x) {
        return parent[x] == x ? x : parent[x] = setOf(parent[x]);
    }
    
    void mergeSetsOf(ull x, ull y) {
        ull a = setOf(x), b = setOf(y);
        if (a == b) return;
        if (size[a] > size[b]) swap(a, b);
        parent[a] = b;
        size[b] += size[a];
    }
    
    bool sameSet(ull x, ull y) { return setOf(x) == setOf(y); }
    
    size_t sizeOfSet(ull s) { return size[s]; }

private:
    vector<ull> parent, size;
};
```

### Disjoint Set Union (Simples)

Parâmetros:

* `n`: intervalo máximo para operações `[1, n]`

Métodos:

* `merge(x, y)`: combina os conjuntos que contém `x` e `y`
* `find(x)`: retorna o representante do conjunto que contém `x`

```c++
struct DSU {
    vector<ll> parent, size;
    DSU(ll n) : parent(n + 1), size(n + 1, 1) {
        iota(all(parent), 0);
    }

    ll find(ll x) {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    }
    
    void merge(ll x, ll y) {
        ull a = find(x), b = find(y);
        if (a == b) return;
        if (size[a] > size[b]) swap(a, b);
        parent[a] = b;
        size[b] += size[a];
    }
};
```

## Algoritmos

### Kruskal

Parâmetros:

* `edges`: grafo representado por vetor de arestas `(peso, u, v)`
* `n`: quantidade máxima de vértices

Retorna: Vetor com a árvore geradora mínima (mst) representado por vetor de arestas
e a soma total de suas arestas

O Grafo precisa ser conectado.

```c++
pair<vector<tuple<ll, ll, ll>>, ll> kruskal(vector<tuple<ll, ll, ll>>& edges, int n) {
    DSU dsu(n);
    vector<tuple<ll, ll, ll>> mst;
    ll edges_sum = 0;
    sort(all(edges));
    for (auto [w, u, v] : edges)
        if (!dsu.sameSet(u, v)) {
            mst.emplace_back(w, u, v);
            dsu.mergeSetsOf(u, v);
            edges_sum += w;
        }
    return { mst, edges_sum };
}
```

### Busca binária

Parâmetros:

* `xs`: vetor ordenado alvo
* `x`: elemento alvo
* `l`: índice de início
* `r`: índice de fim

Retorna: Índice de `x` se encontrado, se não `-1`

Pode ser útil em vez de retornar ```-1```, retornar ```l```

```c++
ll binSearch(vpll& xs, ll x, ll l, ll r) {
    if (l > r) return -1;
    ll m = l + (r - l) / 2;
    if (xs[m].first == x) return m;
    if (xs[m].first < x) l = m + 1;
    else r = m - 1;
    return binSearch(xs, x, l, r);
}
```

### DFS em árvores:

Parâmetros:

* `self`: própria função
* `u`: vértice atual
* `p`: vértice anterior

```c++
auto dfs = [&](auto&& self, ll u, ll p) -> void {
    // processing
    for (auto v : g[u]) if (v != p)
        self(self, v, u);
}; dfs(dfs, 1, -1);
```

### DFS em grafos:

Parâmetros:

* `self`: própria função
* `u`: vértice atual

```c++
vector<bool> vs(g.size());
auto dfs = [&](auto&& self, ll u) -> void {
    vs[u] = true;
    // processing
    for (auto v : g[u]) if (!vs[v])
        self(self, v);
}; dfs(dfs, 1);
```

### Dijkstra

Parâmetros:

* `g`: grafo
* `s`: vértice inicial (menores distâncias em relação à ele)

Retorna: Vetor com as menores distâncias de cada aresta para `s` e vetor de trajetos

```c++
pair<vll, vll> dijkstra(const vector<vpll>& g, ll s) {
    vll ds(g.size(), LONG_LONG_MAX), pre(g.size(), -1);
    priority_queue<pll, vpll, greater<>> pq;
    pq.emplace(0, s); ds[s] = 0;
    while (!pq.empty()) {
        auto [t, u] = pq.top(); pq.pop();
        for (auto [w, v] : g[u])
            if (t + w < ds[v]) {
                pq.emplace(v, t + w);
                ds[v] = t + w;
                pre[v] = u;
            }
    }
    return { ds, pre };
}

vll getPath(const vll& pre, ll s, ll u) {
    vll p { u };
    do {
        p.emplace_back(pre[u]);
        u = pre[u];
    } while (u != s);
    reverse(all(p));
    return p;
}
```

### Binary Lifting:

Parâmetros:

* `n`: quantidade de elementos
* `x`: elemento alvo
* `k`: ordem do ancestral

Retorna: `k` ancestral do elemento `x`

```c++
vector<vector<int>> parent;
const int LOG = 31; // aproximate log of n + 1

void populate(int n) {
    parent.resize(n + 1, vector<int>(LOG));

    // initialize known relationships

    for (int i = 1; i < LOG; ++i)
        for (int j = 1; j < parent.size(); ++j)
            parent[j][i] = parent[ parent[j][i - 1] ][i - 1];
}

// only call after populate
int kthAncestor(int x, int k) {
    int at = x;
    for (int j = 0; j < LOG; j++)
        if (k & 1 << j)
            at = parent[at][j];
    return at;
}
```

Caso queiramos a quantidade de ancestrais de `x`:

```c++
vector<vector<int>> parent;
vector<int> depth;
const int LOG = 31; // aproximate log of n + 1

void populate(int n) {
    parent.resize(n + 1, vector<int>(LOG));

    // initialize known relationships

    // necessary that parent[j] <= j
    for (int j = 1; j < parent.size(); ++j) {
        if (j > 1) depth = depth[parent[j][0]];
        for (int i = 1; i < LOG; ++i)
            parent[j][i] = parent[ parent[j][i - 1] ][i - 1];
    }
}
```

### Divisores:

Retorna: Vetor ordenado com todos os divisores de `x`

```c++
vll divisors(ll x) {
    vll ds {1};
    for (ll i = 2; i * i <= x; ++i)
        if (x % i == 0) {
            ds.emplace_back(i);
            ds.emplace_back(x / i);
        }
    sort(all(ds));
    return ds;
}
```

### Fatoração:

Retorna: Vetor com cada fator primo de `x`

```c++
vll factor(ll x) {
    vll fs;
    for (ll i = 2; i * i <= x; ++i)
        while (x % i == 0) {
            fs.emplace_back(i);
            x /= i;
        }
    if (x > 1) fs.emplace_back(x);
    return fs;
}
```

### Crivo de Eratóstenes:

Retorna: Vetor com todos os primos no intervalo `[1, n]`

```c++
vll sieve(ll n) {
    vll ps;
    vector<bool> is_composite(n);
    for (int i = 2; i < n; ++i) {
        if (!is_composite[i]) ps.emplace_back (i);
        for (int j = 0; j < ps.size() and i * ps[j] < n; ++j) {
            is_composite[i * ps[j]] = true;
            if (i % ps[j] == 0) break;
        }
    }
    return ps;
}
```

### Rotação de Ponto:

Parâmetros:

* `P`: ponto
* `radians`: ângulo em radianos

Retorna: Ponto rotacionado

```c++
pll rotatePoint(const pll& P, double radians) {
    #define x first
    #define y second
    double x = P.x * cos(radians) - P.y * sin(radians);
    double y = P.x * sin(radians) + P.y * cos(radians);
    return {x, y};
}
```

### Orientação de ponto

Parâmetros:

* `A` e `B`: pontos pertencentes à reta
* `P`: ponto que queremos checar orientação

Retorna: Valor que representa a orientação

```c++
// D = 0: P in line
// D > 0: P at left
// D < 0: P at right
ll D(const pll& A, const pll& B, const pll& P) {
    #define x first
    #define y second
    return (A.x * B.y + A.y * P.x + B.x * P.y) - (P.x * B.y + P.y * A.x + B.x * A.y);
}
```
