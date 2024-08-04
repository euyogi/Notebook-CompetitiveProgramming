# Notebook

# Sumário

* Template
* Utils
  * Aritmética modular
  * Direções adjascentes
  * Igualdade flutuante
  * Compressão de coordenadas
  * Próximo maior/menor elemento
  * Fatos
* Estruturas
  * Geometria
    * Reta
    * Segmento
    * Círculo
    * Triângulo
  * Árvores
    * BIT tree
    * Red-Black tree
    * Segment tree
    * Disjoint set union
* Algoritmos
  * Grafos
    * Kruskal (Árvore geradora mínima)
    * DFS
    * BFS 0/1
    * Dijkstra
    * Binary lifting
    * Menor ancestral comum (LCA)
    * Sparse table *
    * Ordenação topológica
  * Outros
    * Busca binária
    * Maior subsequêcia crescente (LIS)
    * Maior subsequência comum (LCS)
  * Matemática
    * Teste de primalidade
    * Divisores
    * Fatoração
    * Crivo de Eratóstenes
  * Geometria
    * Distância entre pontos
    * Rotação de ponto
    * Orientação de ponto
    * Ângulo entre segmentos
    * Ponto contido em segmento
    * Mediatriz

# Main template

```c++
#include <bits/stdc++.h>
using namespace std;

#ifdef LOCAL
#include "dbg.h"
#else
#define dbg(...)
#endif

using ll = long long;
using vll = vector<ll>;
using vvll = vector<vll>;
using pll = pair<ll, ll>;
using vpll = vector<pll>;
using vvpll = vector<vpll>;
using tll = tuple<ll, ll, ll>;
using vtll = vector<tll>;
using ld = long double;
using pld = pair<ld, ld>;

#define all(xs) xs.begin(), xs.end()
#define found(x, xs) (xs.find(x) != xs.end())
#define x first
#define y second

void solve() {

}

signed main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    ll _ts = 1;
    // cin >> _ts;
    while (_ts--) solve();
}
```

# Utils

### Aritmética modular

```c++
static constexpr ll M = 1e9 + 7;
struct Mi {
    ll v;
    Mi() : v(0) {}
    Mi(ll x) : v(x % M) { v += (v < 0) * M; }
    friend bool operator==(Mi a, Mi b) { return a.v == b.v; }
    friend bool operator!=(Mi a, Mi b) { return a.v != b.v; }
    friend ostream& operator<<(ostream& os, Mi a) { return os << a.v; }
    Mi operator+=(Mi b) { return v += b.v - (v + b.v >= M) * M; }
    Mi operator-=(Mi b) { return v -= b.v + (v - b.v < 0) * M; }
    Mi operator*=(Mi b) { return v = v * b.v % M; }
    Mi operator/=(Mi b) & { return *this *= b.inv(); }
    friend Mi operator+(Mi a, Mi b) { return a += b; }
    friend Mi operator-(Mi a, Mi b) { return a -= b; }
    friend Mi operator*(Mi a, Mi b) { return a *= b; }
    friend Mi pow(Mi a, Mi b) {
        return (!b.v ? 1 : pow(a * a, b.v / 2) * (b.v & 1 ? a.v : 1));
    }
    Mi inv() { return pow(*this, M - 2); }
};
```

### 4 direções adjascentes

```c++
vpll ds { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
```

### 8 direções adjascentes

```c++
vpll ds { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 },
          { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 } };
```

### Loop das direções

```c++
for (auto [ox, oy] : ds) {
    ll nx = x + ox, ny = y + oy;
    // processing
}
```

### Igualdade flutuante

```c++
template<typename T, typename S>
bool equals(T a, S b) { return abs(a - b) < 1e-9L; }
```

### Compressão de coordenadas

```c++
void compress(vll& xs) {
    ll c = 0;
    map<ll, ll> mp;
    for (ll x : xs) mp[x] = 0;
    for (auto& p : mp) p.y = c++;
    for (ll& x : xs) x = mp[x];
}
```

### Próximo maior/menor elemento

```c++
// get index of next smallest to the right
// easy to change to be to left or biggest
vll next(xs.size(), xs.size());
stack<ll> prevs;

for (ll i = xs.size() - 1; i >= 0; --i) {
    while (!prevs.empty() and xs[prevs.top()] >= xs[i])
        prevs.pop();
    if (!prevs.empty()) next[i] = prevs.top();
    prevs.emplace(i);
}
```

### Fatos

> `a + b = (a & b) + (a | b)`

> a maior diferença entre dois primos consecutivos `< 10^18` é `1476`

# Estruturas

## Geometria

Provavelmente vai ser necessário definir `x` como `first` e `y`
como `second` e utilizar as funções `dist()` e `equals()`

### Reta

Parâmetros:

* `P`: ponto contido na reta
* `Q`: ponto contido na reta

Métodos:

* `normalizeReal()`: normaliza os coeficientes (reais) da reta
* `normalize()`: normaliza os coeficientes (inteiros) da reta
* `contains(P)`: retorna verdadeiro se `P` está contido na reta, falso caso contrário
* `parallel(r)`: retorna verdadeiro se a reta é paralela a `r`, falso caso contrário
* `orthogonal(r)`: retorna verdadeiro se a reta é perpendicular a `r`, falso caso contrário
* `intersection(r)`: retorna ponto de interseção
* `distance(P)`: retorna a distância de `P` à reta
* `closest(P)`: retorna ponto mais próximo de `P` pertencente à reta

```c++
template<typename T>
struct Line {
    T a, b, c;

    Line(const pair<T, T>& P, const pair<T, T>& Q)
        : a(P.y - Q.y), b(Q.x - P.x), c(P.x * Q.y - Q.x * P.y) {}

    void normalizeReal() { b /= a, c /= a, a = 1; }

    void normalize() {
        if (a < 0 or (a == 0 and b < 0))
            a *= -1, b *= -1, c *= -1;
        T gcd_abc = gcd(a, gcd(b, c));
        a /= gcd_abc, b /= gcd_abc, c /= gcd_abc;
    }

    bool contains(const pair<T, T>& P) {
        return equals(a * P.x + b * P.y + c, 0);
    }

    bool parallel(const Line& r) {
        T det = a * r.b - b * r.a;
        return equals(det, 0) and !(*this == r);
    }

    bool orthogonal(const Line& r) {
        return equals(a * r.a + b * r.b, 0);
    }

    pld intersection(const Line& r) {
        ld det = r.a * b - r.b * a;

        // same or parallel
        if (equals(det, 0)) return {};

        auto x = (-r.c * b + c * r.b) / det;
        auto y = (-c * r.a + r.c * a) / det;
        return { x, y };
    }

    // distance from P to line
    ld distance(const pair<T, T>& P) {
        return abs(a * P.x + b * P.y + c) / hypot(a, b);
    }

    // closest point in line to P
    pld closest(const pair<T, T>& P) {
        ld den = a * a + b * b;
        auto x = (b * (b * P.x - a * P.y) - a * c) / den;
        auto y = (a * (-b * P.x + a * P.y) - b * c) / den;
        return { x, y };
    }

    bool operator==(const Line& r) {
        T k = a ? a : b;
        T s = r.a ? r.a : r.b;
        return equals(a * s, r.a * k) and equals(b * s, r.b * k)
                                      and equals(c * s, r.c * k);
    }
};
```

### Segmento

Parâmetros:

* `P`: ponto extremo do segmento
* `Q`: ponto extremo do segmento

Métodos:

* `contains(P)`: retorna verdadeiro se `P` está contido no segmento, falso caso contrário
* `intersect(r)`: retorna verdadeiro se `r` intersecta com o segmento, falso caso contrário
* `closest(P)`: retorna ponto mais próximo de `P` pertencente ao segmento

```c++
template<typename T>
struct Segment {
    pair<T, T> A, B;

    Segment(const pair<T, T>& P, const pair<T, T>& Q)
        : A(P), B(Q) {}

    bool contains(const pair<T, T>& P) {
        auto dAB = dist(A, B), dAP = dist(A, P), dPB = dist(P, B);
        return equals(dAP + dPB, dAB);
    }

    bool intersect(const Segment& r) {
        auto d1 = D(A, B, r.A), d2 = D(A, B, r.B);
        auto d3 = D(r.A, r.B, A), d4 = D(r.A, r.B, B);

        if ((equals(d1, 0) and contains(r.A)) or
            (equals(d2, 0) and contains(r.B)))
            return true;

        if ((equals(d3, 0) and r.contains(A)) or
            (equals(d4, 0) and r.contains(B)))
            return true;

        return (d1 * d2 < 0) and (d3 * d4 < 0);
    }

    // closest point in segment to P
    pair<T, T> closest(const pair<T, T>& P) {
        Line<T> r(A, B);
        auto Q = r.closest(P);
        auto distA = dist(A, P), distB = dist(B, P);
        if (this->contains(Q)) return Q;
        if (distA <= distB) return A;
        return B;
    }
};
```
### Círculo

Parâmetros:

* `P`: ponto central
* `r`: raio

Métodos:

* `area()`: retorna área
* `perimeter()`: retorna perímetro
* `arc(radians)`: retorna comprimento do arco
* `chord(radians)`: retorna comprimento da corda
* `sector(radians)`: retorna área do setor
* `segment(radians)`: retorna área do segmento
* `position(P)`: retorna valor que representa a posição do ponto
* `intersection(c)`: retorna pontos de interseção com círculo
* `intersection(P, Q)`: retorna pontos de interseção com reta `PQ`

```c++
enum PointPosition { IN, ON, OUT };

template<typename T>
struct Circle {
    pair<T, T> C;
    T r;

    Circle(const pair<T, T>& P, T r) : C(P), r(r) {}

    ld area() { return M_PI * r * r; }
    ld perimeter() { return 2.0L * M_PI * r; }
    ld arc(ld radians) { return radians * r; }

    ld chord(ld radians) {
        return 2.0L * r * sin(radians / 2.0L);
    }

    ld sector(ld radians) {
        return (radians * r * r) / 2.0L;
    }

    ld segment(ld radians) {
        auto c = chord(radians);
        auto s = (r + r + c) / 2.0L;
        auto t = sqrt(s) * sqrt(s - r) * sqrt(s - r) * sqrt(s - c);
        return sector(radians) - t;
    }

    PointPosition position(const pair<T, T>& P) {
        auto d = dist(P, C);
        return equals(d, r) ? ON : (d < r ? IN : OUT);
    }

    vector<pair<T, T>> intersection(const Circle& c) {
        auto d = dist(c.C, C);

        // no intersection or same
        if (d > c.r + r or d < abs(c.r - r) or
            (equals(d, 0) and equals(c.r, r)))
            return {};

        auto a = (c.r * c.r - r * r + d * d) / (2.0L * d);
        auto h = sqrt(c.r * c.r - a * a);
        auto x = c.C.x + (a / d) * (C.x - c.C.x);
        auto y = c.C.y + (a / d) * (C.y - c.C.y);
        pld P1, P2;
        P1.x = x + (h / d) * (C.y - c.C.y);
        P1.y = y - (h / d) * (C.x - c.C.x);
        P2.x = x - (h / d) * (C.y - c.C.y);
        P2.y = y + (h / d) * (C.x - c.C.x);
        return P1 == P2 ? vector<pair<T, T>> { P1 } :
                          vector<pair<T, T>> { P1, P2 };
    }

    // circle at origin
    vector<pair<T, T>>
    intersection(const pair<T, T>& P, const pair<T, T>& Q) {
        ld a(P.y - Q.y), b(Q.x - P.x), c(P.x * Q.y - Q.x * P.y);
        auto x0 = -a * c / (a*a + b*b), y0 = -b * c / (a*a + b*b);
        if (c*c > r*r * (a*a + b*b) + 1e-9L) return {};
        if (equals(c*c, r*r * (a*a + b*b))) return { { x0, y0 } };
        auto d = r*r - c*c / (a*a + b*b);
        auto mult = sqrt(d / (a*a + b*b));
        auto ax = x0 + b * mult;
        auto bx = x0 - b * mult;
        auto ay = y0 - a * mult;
        auto by = y0 + a * mult;
        return { { ax, ay }, { bx, by } };
    }
};
```

### Triângulo

Parâmetros:

* `P`: primeiro ponto
* `Q`: segundo ponto
* `R`: terceiro ponto

Métodos:

* `area()`: retorna área
* `perimeter()`: retorna perímetro
* `sidesClassification()`: retorna valor que representa a classificação do triângulo
* `anglesClassification()`: retorna valor que representa a classificação do triângulo
* `barycenter()`: retorna ponto de interseção entre as medianas
* `circumradius`: retorna valor do raio da circunferência circunscrita
* `circumcenter`: retorna ponto de interseção entre as as retas perpendiculares que interceptam nos pontos médios
* `inradius()`: retorna valor do raio da circunferência inscrita
* `incenter(c)`: retorna ponto de interseção entre as bissetrizes
* `orthocenter(P, Q)`: retorna ponto de interseção entre as alturas

```c++
enum Class { EQUILATERAL, ISOSCELES, SCALENE };
enum Angles { RIGHT, ACUTE, OBTUSE };

template <typename T>
struct Triangle {
    pair<T, T> A, B, C;
    T a, b, c;

    Triangle(pair<T, T> P, pair<T, T> Q, pair<T, T> R)
        : A(P), B(Q), C(R), a(dist(A, B)), b(dist(B, C)), c(dist(C, A)) {}

    T area() {
        T det = (A.x * B.y + A.y * C.x + B.x * C.y) -
                (C.x * B.y + C.y * A.x + B.x * A.y);
        if (is_floating_point_v<T>) return 0.5L * abs(det);
        return abs(det);
    }

    ld perimeter() { return a + b + c; }

    Class sidesClassification() {
        if (equals(a, b) and equals(b, c)) return EQUILATERAL;
        if (equals(a, b) or equals(a, c) or equals(b, c)) return ISOSCELES;
        return SCALENE;
    }

    Angles anglesClassification() {
        auto alpha = acos((a * a - b * b - c * c) / (-2.0L * b * c));
        auto beta = acos((b * b - a * a - c * c) / (-2.0L * a * c));
        auto gamma = acos((c * c - a * a - b * b) / (-2.0L * a * b));
        auto right = M_PI / 2.0L;
        if (equals(alpha, right) || equals(beta, right)
                                 || equals(gamma, right))
            return RIGHT;
        if (alpha > right || beta > right || gamma > right) return OBTUSE;
        return ACUTE;
    }

    pld barycenter() {
        auto x = (A.x + B.x + C.x) / 3.0L;
        auto y = (A.y + B.y + C.y) / 3.0L;
        return {x, y};
    }

    ld circumradius() { return (a * b * c) / (4.0L * area()); }

    pld circumcenter() {
        ld D = 2 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y));
        T A2 = A.x * A.x + A.y * A.y;
        T B2 = B.x * B.x + B.y * B.y;
        T C2 = C.x * C.x + C.y * C.y;
        auto x = (A2 * (B.y - C.y) + B2 * (C.y - A.y) + C2 * (A.y - B.y)) / D;
        auto y = (A2 * (C.x - B.x) + B2 * (A.x - C.x) + C2 * (B.x - A.x)) / D;
        return {x, y};
    }

    ld inradius() { return (2 * area()) / perimeter(); }

    pld incenter() {
        auto P = perimeter();
        auto x = (a * A.x + b * B.x + c * C.x) / P;
        auto y = (a * A.y + b * B.y + c * C.y) / P;
        return {x, y};
    }

    pld orthocenter() {
        Line<T> r(A, B), s(A, C);
        Line<T> u{r.b, -r.a, -(C.x * r.b - C.y * r.a)};
        Line<T> v{s.b, -s.a, -(B.x * s.b - B.y * s.a)};
        ld det = u.a * v.b - u.b * v.a;
        auto x = (-u.c * v.b + v.c * u.b) / det;
        auto y = (-v.c * u.a + u.c * v.a) / det;
        return {x, y};
    }
};
```

## Árvores

### BIT tree

Parâmetros:

* `n`: intervalo máximo para as operações `[1, n]`

Métodos:

* `rangeAdd(i, j, x)`: soma `x` em cada elemento no intervalo `[i, j]`
* `rangeQuery(i, j)`: retorna a soma do intervalo `[i, j]`

```c++
class BIT {
public:
    BIT(ll n) : bt1(n+1), bt2(n+1) {}

    void rangeAdd(ll i, ll j, ll x) {
    	add(i, x, bt1);           add(j + 1, -x, bt1);
    	add(i, x * (i - 1), bt2); add(j + 1, -x * j, bt2);
    }

    ll rangeQuery(ll i, ll j) {
    	return rsq(j, bt1) * j           - rsq(j, bt2) -
    	      (rsq(i - 1, bt1) * (i - 1) - rsq(i - 1, bt2));
    }

private:
    void add(ll i, ll x, vll& bt) {
        for (; i < bt.size(); i += i & -i) bt[i] += x;
    }

    ll rsq(ll i, vll& bt) {
        ll sum = 0;
        for (; i >= 1 i -= i & -i) sum += bt[i];
        return sum;
    }

    vll bt1, bt2;
};
```

### BIT tree (Simples)

Parâmetros:

* `n`: intervalo máximo para as operações `[1, n]`

Métodos:

* `add(i, x)`: soma `x` no índice `i`
* `rsq(i)`: retorna a soma do intervalo `[1, i]`

```c++
struct BIT {
    vll bt
    BIT(ll n) : bt(n+1) {}

    void add(ll i, ll x) {
        for (; i < bt.size(); i += i & -i) bt[i] += x;
    }

    ll rsq(ll i) {
        ll sum = 0;
        for (; i >= 1; i -= i & -i) sum += bt[i];
        return sum;
    }
};
```

### Red-Black tree

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

### Segment tree (Simples)

Parâmetros:

* `n`: intervalo máximo para operações `[0, n)`

Métodos:

* `setQuery(i, j, x, l, r, node)`: super função (set/query).
  `i` e `j` é o intervalo da nossa ação. `x` é um argumento
  opcional que é passado somente quando queremos setar um valor.
  `l` e `r` é o intervalo que o nó representa. `node` é o nó
  atual. esses três últimos argumentos não são passados nunca.
  se passamos `x` (set) não deve-se utilizar o valor retornado,
  caso contrário (query), retorna valor correspondente à
  funcionalidade codificada de acordo com o tipo passado como
  template

```c++
template<typename T>
class SegTree {
public:
    SegTree(ll n) {
        while (__builtin_popcount(n) != 1) ++n;
        (*this).n = n;
        seg.resize(2 * n, DEF);
    }

    T setQuery(ll i, ll j, T x = LONG_LONG_MIN, ll l = 0, ll r = 0, ll node = 0) {
        if (not node) r = n - 1, node = 1;
        if (i <= l and r <= j) {
            // change accordingly
            if (x != LONG_LONG_MIN) seg[node] = x;
            return seg[node];
        }
        if (j < l or i > r) return DEF;
        ll m = l + (r - l) / 2;
        // change accordingly
        ll sum = (setQuery(i, j, x, l, m, node * 2) +
                  setQuery(i, j, x, m + 1, r, node * 2 + 1));
        seg[node] = (seg[2 * node] + seg[2 * node + 1]);
        return sum;
    }

private:
    const T DEF = 0; // change accordingly
    vector<T> seg;
    ll n;
};
```
### Disjoint set union

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
    DSU(ll n) : parent(n + 1), size(n + 1, 1) {
        iota(all(parent), 0);
    }

    ll setOf(ll x) {
        return parent[x] == x ? x : parent[x] = setOf(parent[x]);
    }

    void mergeSetsOf(ll x, ll y) {
        ll a = setOf(x), b = setOf(y);
        if (size[a] > size[b]) swap(a, b);
        parent[a] = b;
        if (a != b) size[b] += size[a];
    }

    bool sameSet(ll x, ll y) { return setOf(x) == setOf(y); }
    size_t sizeOfSet(ll s) { return size[s]; }

private:
    vll parent, size;
};
```

### Disjoint set union (Simples)

Parâmetros:

* `n`: intervalo máximo para operações `[1, n]`

Métodos:

* `merge(x, y)`: combina os conjuntos que contém `x` e `y`
* `find(x)`: retorna o representante do conjunto que contém `x`

```c++
struct DSU {
    vll parent, size;
    DSU(ll n) : parent(n + 1), size(n + 1, 1) {
        iota(all(parent), 0);
    }

    ll find(ll x) {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    }

    void merge(ll x, ll y) {
        ll a = find(x), b = find(y);
        if (size[a] > size[b]) swap(a, b);
        parent[a] = b;
        if (a != b) size[b] += size[a];
    }
};
```

# Algoritmos

## Grafos

### Kruskal

Parâmetros:

* `edges`: grafo representado por vetor de arestas `(peso, u, v)`
* `n`: quantidade máxima de vértices

Retorna: Vetor com a árvore geradora mínima (mst), se o grafo for conectado,
representado por vetor de arestas e a soma total de suas arestas

```c++
pair<vtll, ll> kruskal(vtll& edges, int n) {
    DSU dsu(n);
    vtll mst;
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

### DFS em árvores

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

### DFS em grafos

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

### BFS 0/1

Parâmetros:

* `g`: grafo alvo
* `s`: vértice inicial (menores distâncias em relação à ele)

Retorna: Vetor com as menores distâncias de cada aresta para `s`

```c++
vll bfs01(const vvpll& g, ll s) {
    vll ds(g.size(), LONG_LONG_MAX);
    deque<ll> dq;
    dq.emplace(s); ds[s] = 0;
    while (!dq.empty()) {
        ll u = pq.front(); pq.pop();
        for (ll v : g[u])
            if (ds[u] + w < ds[v]) {
                ds[v] = ds[u] + w;
                if (w == 1) dq.emplace_back(v)
                else dq.emplace_front(v);
            }
    }
    return ds;
}
```

### Dijkstra

Parâmetros:

* `g`: grafo alvo
* `s`: vértice inicial (menores distâncias em relação à ele)

Retorna: Vetor com as menores distâncias de cada aresta para `s` e vetor de trajetos

```c++
pair<vll, vll> dijkstra(const vvpll& g, ll s) {
    vll ds(g.size(), LONG_LONG_MAX), pre(g.size(), -1);
    priority_queue<pll, vpll, greater<>> pq;
    pq.emplace(0, s); ds[s] = 0;
    while (!pq.empty()) {
        auto [t, u] = pq.top(); pq.pop();
        if (ds[u] < u) continue;
        for (auto& [w, v] : g[u])
            if (t + w < ds[v]) {
                ds[v] = t + w, pre[v] = u;
                pq.emplace(v, t + w);
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

### Binary lifting

Parâmetros:

* `n`: quantidade de vértices/elementos
* `x`: vértice/elemento alvo
* `k`: ordem do ancestral

Retorna: `k` ancestral do vértice/elemento `x`

```c++
const int LOG = 31; // aproximate log of n, + 1

vector<vector<int>> parent;
vector<int> depth;

// if isn't a graph delete parameter g
void populate(int n, vector<vector<int>>& g) {
    parent.resize(n + 1, vector<int>(LOG));
    depth.resize(n + 1);

    // initialize known relationships (e.g.: dfs if it's a graph)

    // parent[1][0] = 1;
    // auto dfs = [&](auto&& self, int u, int p) -> void {
    //     for (ll v : g[u]) if (v != p) {
    //         parent[v][0] = u;
    //         depth[v] = depth[u] + 1;
    //         self(self, v, u);
    //     }
    // }; dfs(dfs, 1, -1);

    for (ll i = 1; i < LOG; ++i)
        for (ll j = 1; j <= n; ++j)
            parent[j][i] = parent[ parent[j][i - 1] ][i - 1];
}

int kthAncestor(int u, int k) {
    // if (k > depth[u]) return -1; // no kth ancestor
    for (int i = 0; i < LOG; ++i)
        if (k & (1 << i))
            u = parent[u][i];
    return u;
}
```

### Menor ancestral comum (LCA)

Parâmetros:

* `u`: primeiro vértice/elemento
* `v`: segundo vértice/elemento

Retorna: Menor ancestral comum entre `u` e `v`


```c++
int LCA(int u, int v) {
    if (depth[u] < depth[v]) swap(u, v);
    int k = depth[u] - depth[v];
    u = kthAncestor(u, k);
    if (u == v) return u;
    for (int i = LOG - 1; i >= 0; --i)
        if (parent[u][i] != parent[v][i]) {
            u = parent[u][i];
            v = parent[v][i];
        }
    return parent[u][0];
}
```

### Ordenação topológica

Parâmetros:

* `g`: grafo alvo

Retorna: Vetor com a ordenação topológica do grafo, se houver ciclo retorna vetor vazio

```c++
vll topologicalSort(vvll& g) {
    vll degree(g.size()), res;
    for (int i = 1; i < g.size(); ++i)
        for (auto u : g[i])
            ++degree[u];

    // lower values bigger priorities
    priority_queue<ll, vll, greater<>> pq;
    for (int i = 1; i < degree.size(); ++i)
        if (degree[i] == 0) pq.emplace(i);

    while (!pq.empty()) {
        ll u = pq.top();
        pq.pop();
        res.emplace_back(u);
        for (ll v : g[u])
            if (--degree[v] == 0)
                pq.emplace(v);
    }

    if (res.size() != g.size())
        return {}; // cycle

    return res;
}
```

## Outros

### Busca binária

Parâmetros:

* `xs`: vetor ordenado alvo
* `x`: elemento alvo
* `l`: índice de início
* `r`: índice de fim

Retorna: Índice de `x` se encontrado, se não `-1`

Pode ser útil em vez de retornar ```-1```, retornar ```l```

```c++
ll binSearch(vll& xs, ll x, ll l, ll r) {
    if (l > r) return -1;
    ll m = l + (r - l) / 2;
    if (xs[m] == x) return m;
    if (xs[m] < x) l = m + 1;
    else r = m - 1;
    return binSearch(xs, x, l, r);
}
```

### Maior subsequência crescente (LIS)

Parâmetros:

* `xs`: sequência alvo

Retorna: Par com o tamanho da maior subsequência crescente e o último elemento dela

```c++
pll LIS(vll& xs) {
    vll ss;
    for (ll x : xs) {
        auto it = lower_bound(all(ss), x);
        if (it == ss.end()) ss.emplace_back(x);
        else *it = x;
    }
    return { ss.size(), ss.back() };
}
```

### Maior subsequência comum (LCS)

Parâmetros:

* `xs`: primeira sequência
* `ys`: segunda sequência

Retorna: Tamanho da maior subsequência comum

```c++
template<typename T>
ll LCS(vector<T>& xs, vector<T>& ys) {
    vvll dp(xs.size() + 1, vll(ys.size() + 1));
    for (ll i = 1; i <= xs.size(); ++i)
        for (ll j = 1; j <= ys.size(); ++j) {
            if (xs[i - 1] == ys[j - 1])
                dp[i][j] = 1 + dp[i - 1][j - 1];
            else
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j]);
        }
    return dp.back().back();
}
```

## Matemática

### Teste de primalidade

Retorna: Verdadeiro se `x` é primo, falso caso contrário
```c++
bool isPrime(ll x) {
    if (i == 1) return false;

    for (ll i = 2; i * i <= x; ++i)
        if (x % i == 0) return false;

    return true;
}
```

### Divisores

Retorna: Vetor ordenado com todos os divisores de `x`

```c++
vll divisors(ll x) {
    vll ds {1};
    for (ll i = 2; i * i <= x; ++i)
        if (x % i == 0) {
            ds.emplace_back(i);
            if (i * i != x)
                ds.emplace_back(x / i);
        }
    return ds;
}
```

### Fatoração

Retorna: Vetor com cada fator primo de `x`

```c++
vll factors(ll x) {
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

### Crivo de Eratóstenes

Retorna: Vetor com todos os primos no intervalo `[1, n]` e vetor de menor fator primo

```c++
pair<vll, vll> sieve(ll n) {
    vll ps, spf(n + 1);
    for (ll i = 2; i <= n; i++)
        if (!spf[i]) {
            spf[i] = i;
            ps.emplace_back(i);
            for (ll j = i * i; j <= n; j += i)
                if (!spf[j]) spf[j] = i;
        }
    return { ps, spf };
}
```

Exemplo de fatoração em O(log N) com o vetor de menor fator primo:

```c++
auto [ps, spf] = sieve(42);
for (ll i = 0; i < n; i++) {
    ll v;
    cin >> v;
    while (v != 1) {
        cout << spf[v] << ' ';
        v /= spf[v];
    }
    cout << '\n';
}
```

## Geometria

### Distância entre pontos

Retorna: Distância entre os pontos `P` e `Q`

```c++
template<typename T>
ld dist(const pair<T, T>& P, const pair<T, T>& Q) {
    return hypot(P.x - Q.x, P.y - Q.y);
}
```

### Rotação de ponto

Retorna: Ponto rotacionado

```c++
template<typename T>
pld rotate(const pair<T, T>& P, ld radians) {
    auto x = cos(radians) * P.x - sin(radians) * P.y;
    auto y = sin(radians) * P.x + cos(radians) * P.y;
    return { x, y };
}
```

### Orientação de ponto

Retorna: Valor que representa a orientação do ponto

```c++
// D = 0: P in line
// D > 0: P at left
// D < 0: P at right
template<typename T>
T D(const pair<T, T>& A, const pair<T, T>& B, const pair<T, T>& P) {
    return (A.x * B.y + A.y * P.x + B.x * P.y) -
           (P.x * B.y + P.y * A.x + B.x * A.y);
}
```

### Ângulo entre segmentos

Retorna: Ângulo entre segmentos

```c++
// smallest angle between segments PQ and RS
template<typename T> ld
angle(const pair<T, T>& P, const pair<T, T>& Q,
      const pair<T, T>& R, const pair<T, T>& S) {
    T ux = P.x - Q.x, uy = P.y - Q.y;
    T vx = R.x - S.x, vy = R.y - S.y;
    T num = ux * vx + uy * vy;

    // degenerate segment: den = 0
    auto den = hypot(ux, uy) * hypot(vx, vy);
    return acos(num / den);
}
```

### Ponto contido em segmento

Retorna: Verdadeiro se o ponto está contido no segmento, falso caso contrário

```c++
// P in segment AB
template<typename T>
bool contains(const pair<T, T>& A, const pair<T, T>& B, const pair<T, T>& P) {
    auto xmin = min(A.x, B.x), xmax = max(A.x, B.x);
    auto ymin = min(A.y, B.y), ymax = max(A.y, B.y);

    if (P.x < xmin || P.x > xmax || P.y < ymin || P.y > ymax)
        return false;

    return equals((P.y - A.y) * (B.x - A.x), (P.x - A.x) * (B.y - A.y));
}
```

### Mediatriz

Retorna: Reta mediatriz ao segmento `PQ`

```c++
template<typename T>
Line<T> perpendicularBisector(const pair<T, T>& P, const pair<T, T>& Q) {
    auto a = 2 * (Q.x - P.x);
    auto b = 2 * (Q.y - P.y);
    auto c = (P.x * P.x + P.y * P.y) - (Q.x * Q.x + Q.y * Q.y);
    return { a, b, c };
}
```
