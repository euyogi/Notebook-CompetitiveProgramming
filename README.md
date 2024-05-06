# Template

```c++
#include <bits/stdc++.h>

#ifdef DEBUG
#include "dbg.h"
#else
#define dbg(...)
#endif

using namespace std; using ll = long long; using ull = unsigned long long; using pll = pair<ll, ll>; using vll = vector<ll>; using vvll = vector<vll>; using vpll = vector<pll>; using Point = pll;
#define all(vs) vs.begin(), vs.end()
#define found(x, xs) (xs.find(x) != xs.end())

void tomaraQuePasse() {

}

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int testes = 1;
    // cin >> testes; cin.ignore();
    while (testes--) tomaraQuePasse();
}
```

# Lista de direções

4 direções adjascentes:

```c++
pll ds[4][2] { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };
```

8 direções adjascentes:

```c++
pll ds[8][2] { {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };
```

Loop das direções:

```c++
for (auto [ix, iy] : ds) {
    int nx = x + ix, ny = y + iy;
}
```

# Igualdade flutuante

```c++
template<typename T>
bool equals(T a, T b) {
    constexpr double EPS { 1e-9 };

    return std::is_floating_point<T>::value ?  fabs(a-b) < EPS : a == b;
}
```

# GCD

Para versões < C++ 17.

```c++
int gcd_(ll a, ll b) {
    while (b) {
        a %= b;
        swap(a, b);
    }

    return a;
}
```

# Divisores

```c++
vll divisors(ll n) {
    vll ans {1};
    for (ll i = 2; i*i <= n; ++i)
        if (n % i == 0) {
            ans.emplace_back(i);
            ans.emplace_back(n/i);
        }

    sort(all(ans)); // Comentar caso ordem não importe
    return ans;
}
```

# Base para busca binária

Vetor precisa estar ordenado.

Se em vez de retornar ```xs.end()```, retornar ```l```, teremos o equivalente a um upper_bound.

```c++
auto binSearch(vll& xs, ll x, size_t l, size_t r) {
    if (l > r) return xs.end();

    size_t m = (l + r) / 2;
    if (xs[m] == x) return xs.begin() + m;

    l = (xs[m] < x ? m + 1 : l);
    r = (xs[m] > x ? m - 1 : r);
    return binSearch(xs, x, l, r);
}
```

# Base para DFS em árvore

Árvore = (Grafo não direcionado, conectado e acíclico).

Note que chamei o grafo de g, e iniciei a dfs no vértice 1.

Grafo consiste de uma lista de adjacências não ponderada.

```c++
auto dfs = [&](auto&& dfs, ll c, ll p) -> void {
    // Processa c
    for (auto n : g[c]) if (n != p)
            dfs(dfs, n, c);
}; dfs(dfs, 1, -1);
```

DFS em grafos que não sejam árvores será necessário uma forma de guardar os nós visitados para não voltar neles.

# BIT Tree

Somar valores em intervalos.

n = valor máximo do intervalo (vai ir de 1 até n inclusivo).

```c++
class BIT {
public:
    BIT(size_t n) : bt1(n+1), bt2(n+1) {}

    ll rangeQuery(size_t i, size_t j) {
    	return rsq(j, bt1) * j       - rsq(j, bt2) -
    	      (rsq(i-1, bt1) * (i-1) - rsq(i-1, bt2));
    }

    void rangeAdd(size_t i, size_t j, ll x) {
    	add(i, x, bt1);         add(j+1, -x, bt1);
    	add(i, x * (i-1), bt2); add(j + 1, -x * j, bt2);
    }

private:
    ll rsq(ll i, vll& bt) {
        ll sum = 0;

        while (i >= 1) {
            sum += bt[i];
            i -= i & (-i);
        }

        return sum;
    }

    void add(size_t i, ll x, vll& bt) {
        while (i < bt.size()) {
            bt[i] += x;
            i += i & (-i);
    	}
    }

    vll bt1, bt2;
};
```

# Red-Black Tree

Armazenar valores ordenadamente e conseguir acessar
o índice do primeiro elemento maior ou igual a x.

```c++
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace __gnu_pbds;

typedef tree<ll, null_type, less_equal<>,
rb_tree_tag,tree_order_statistics_node_update> set_t;
```

# Dijkstra

Retorna a menor distância de cada aresta para uma principal e respectivos caminhos.

Grafo consiste de uma lista de adjacências com pares (vértice, peso).

```c++
pair<vll, vll> dijkstra(const vector<vpll>& g, ll s) {
    vll ds(g.size(), LONG_LONG_MAX), pre(g.size(), -1);
    priority_queue<pll, vpll, greater<>> pq;
    
    pq.emplace(s, 0);
    ds[s] = 0;
    
    while (!pq.empty()) {
        auto [c, t] = pq.top();
        pq.pop();
    
        for (auto [n, w] : g[c])
            if (t + w < ds[n]) {
                pq.emplace(n, t + w);
                ds[n] = t + w;
                pre[n] = c;
            }
    }
    
    return { ds, pre };
}
```

Retorna o caminho: (Não funcionará se o caminho não existe ou se é para o mesmo elemento).

```c++
vll getPath(const vll& pre, ll s, ll t) {
    vll p { t };
    do {
        p.emplace_back(pre[t]);
        t = pre[t];
    } while (t != s);
    reverse(all(p));
    return p;
}
```

# Disjoint Set Union

Representar conjuntos de elementos, conseguir saber de qual conjunto
um elemento é e conseguir saber quantos elementos existem nesse conjunto.
Útil também para identificar ciclos.

n = valor máximo dos elementos (serão criados n+1 conjuntos, do 0 ao n).

```c++
class DSU {
public:
    DSU(size_t n) : parent(n+1), size(n+1, 1) {
        iota(all(parent), 0);
    }

    ll setOf(ull x) {
        return parent[x] == x ? x : parent[x] = setOf(parent[x]);
    }

    bool sameSet(ull x, ull y) { return setOf(x) == setOf(y); }

    void mergeSetsOf(ull x, ull y) {
        ull a = setOf(x), b = setOf(y);
        if (a == b) return;
        if (size[a] > size[b]) swap(a, b);
        parent[a] = b;
        size[b] += size[a];
    }

    size_t sizeOfSet(ull i) { return size[i]; }

private:
    vector<ull> parent, size;
};
```

# Kruskal

Retorna a árvore de extensão mínima (mst) e a soma dos pesos de todas suas arestas.

O Grafo precisa ser conectado.

Grafo consiste de uma lista de arestas com triplas (peso, a, b).

```c++
pair<vector<tuple<ll, ll, ll>>, ll> kruskal(vector<tuple<ll, ll, ll>>& edges, int n) {
    DSU dsu(n);
    vector<tuple<ll, ll, ll>> mst;
    ll ws_sum = 0;

    sort(all(edges));
    for (auto [w, a, b] : edges)
        if (!dsu.sameSet(a, b)) {
            mst.emplace_back(w, a, b);
            dsu.mergeSetsOf(a, b);
            ws_sum += w;
        }

    return { mst, ws_sum };
}
```

# Geometria

Rotacionar Ponto:

```c++
Point rotatePoint(const Point& P, double angleRadians) {
    #define x first
    #define y second

    double x = P.x * cos(angleRadians) - P.y * sin(angleRadians);
    double y = P.x * sin(angleRadians) + P.y * cos(angleRadians);

    return {x, y};
}
```

Checar orientação do ponto P em relação à reta AB:

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

Checar ângulo entre as retas AB e CD:

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

A linha criada terá seus coeficientes normalizados.

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
