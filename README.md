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
  * Outros
    * Maior subsequência comum (LCS)
    * Maior subsequência crescente (LIS)
    * Soma de prefixo 2D
    * Soma de prefixo 3D
  * Matemática
    * Coeficiente binomial
    * Conversão de base
    * Crivo de Eratóstenes
    * Divisores
    * Fatoração
    * GCD
    * LCM
    * Quantidade de divisores
  * Strings
    * Z-Function
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
  * Próximo maior/menor elemento
  * Soma de todos os intervalos

# Template

```c++
#include <bits/stdc++.h>
using namespace std;

#ifdef croquete  // BEGIN TEMPLATE ----------------------|
#include "dbg/dbg.h"
#else
#define dbg(...)
#endif
#define ll    long long
#define vll   vector<ll>
#define vvll  vector<vll>
#define pll   pair<ll, ll>
#define vpll  vector<pll>
#define vvpll vector<vpll>
#define tll   tuple<ll, ll, ll>
#define vtll  vector<tll>
#define pd    pair<double, double>
#define vb    vector<bool>
#define all(xs)      xs.begin(), xs.end()
#define found(x, xs) (xs.find(x) != xs.end())
#define rep(i, a, b) for (ll i = (a); i < (ll)(b); ++i)
#define per(i, a, b) for (ll i = (a); i >= (ll)(b); --i)
#define x     first
#define y     second
#define eb    emplace_back
#define cinj  (cin.iword(0) = 1, cin)
#define coutj (cout.iword(0) = 1, cout)
template <typename T>  // read vector
istream& operator>>(istream& in, vector<T>& xs) {
    assert(!xs.empty());
    rep(i, in.iword(0), xs.size()) in >> xs[i];
    in.iword(0) = 0;
    return in;
}
template <typename T>  // print vector
ostream& operator<<(ostream& os, vector<T>& xs) {
    rep(i, os.iword(0), xs.size() - 1) os << xs[i] << ' ';
    os.iword(0) = 0;
    return os << xs.back();
}  // END TEMPLATE --------------------------------------|

void pqnpassa() {}

signed main() {
    cin.tie(0)->sync_with_stdio(0);
    ll t = 1;
    // cin >> t;
    while (t--) pqnpassa();
}
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
    void C(T x, int n = 4)  { cerr << fixed << "\033[9" << n << 'm' << x << "\033[m"; }
    void p(char x)          { C("\'" + string({x}) + "\'", 3); }
    void p(string x)        { C("\"" + x + "\"", 3); }
    void p(bool x)          { x ? C('T', 2) : C('F', 1); }
    void p(vector<bool> xs) { for (bool x : xs) p(x); }
    static int m = 0;
    template <typename T>
    void p(T x) {
        int i = 0;
        if constexpr (requires { begin(x); }) { // nested iterable
            C('{');
            if (size(x) && requires { begin(*begin(x)); }) {
                cerr << '\n';
                m += 2;
                for (auto y : x)
                    cerr << string(m, ' ') << setw(2) << left << i++, p(y), cerr << '\n';
                cerr << string(m -= 2, ' ');
            } else // normal iterable
                for (auto y : x) i++ ? C(',') : C(""), p(y);
            C('}');
        } else if constexpr (requires { x.pop(); }) { // stacks, queues
            C('{');
            while (!x.empty()) {
                if (i++) C(',');
                if constexpr (requires { x.top(); }) p(x.top());
                else p(x.front());
                x.pop();
            }
            C('}');
        } else if constexpr (requires { get<0>(x); }) { // pairs, tuples
            C('(');
            apply([&](auto... args) { ((i++ ? C(',') : C(""), p(args)), ...); }, x);
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

Parâmetros:

* `(P, Q, R, S)`: Pontos extremos dos segmentos `PQ` e `RS`

Retorna: Menor ângulo entre os segmentos em radianos

```c++
template <typename T>
double angle(const pair<T, T>& P, const pair<T, T>& Q,
             const pair<T, T>& R, const pair<T, T>& S) {
    T ux = P.x - Q.x, uy = P.y - Q.y;
    T vx = R.x - S.x, vy = R.y - S.y;
    T num = ux * vx + uy * vy;

    // degenerate segment: den = 0
    double den = hypot(ux, uy) * hypot(vx, vy);
    return acos(num / den);
}
```

### Distância entre pontos

Parâmetros:

* `(P, Q)`: Pontos alvo

Retorna: Distância entre os pontos

```c++
template <typename T, typename S>
double dist(const pair<T, T>& P, const pair<S, S>& Q) {
    return hypot(P.x - Q.x, P.y - Q.y);
}
```

### Envoltório convexo

Parâmetros:

* `(ps)`: Vetor de pontos (representando polígono, sem repetir ponto inicial)

Retorna: Vetor dos pontos do envoltório convexo (repetindo ponto inicial,
sentido anti-horário)

```c++
template <typename T>
vector<pair<T, T>> makeHull(vector<pair<T, T>>& ps) {
    vector<pair<T, T>> hull;
    for (auto& p : ps) {
        auto sz = hull.size();          // if want collinear < 0
        while (sz >= 2 and D(hull[sz - 2], hull[sz - 1], p) <= 0) {
            hull.pop_back();
            sz = hull.size();
        }
        hull.eb(p);
    }
    return hull;
}

template <typename T>
vector<pair<T, T>> monotoneChain(vector<pair<T, T>> ps) {
    vector<pair<T, T>> lower, upper;
    sort(all(ps));
    lower = makeHull(ps);
    reverse(all(ps));
    upper = makeHull(ps);
    lower.pop_back();
    lower.emplace(lower.end(), all(upper));
    return lower;
}
```

### Orientação de ponto

Parâmetros:

* `(A, B)`: Pontos extremos do segmento `AB`
* `(P)`: Ponto alvo

Retorna: Orientação do ponto em relação ao segmento

Adendos:

* `D = 0`: `P` Colinear
* `D > 0`: `P` À esquerda
* `D < 0`: `P` À direita

```c++
template <typename T>
T D(const pair<T, T>& A, const pair<T, T>& B, const pair<T, T>& P) {
    return (A.x * B.y + A.y * P.x + B.x * P.y) - (P.x * B.y + P.y * A.x + B.x * A.y);
}
```

Parâmetros:

* `(P, Q, O)`: Pontos alvo, `O` é o centro de referência

Retorna: Se o ponto `P` vem antes do ponto `Q` no sentido anti-horário em relação ao
centro

```c++
template <typename T>
bool ccw(pair<T, T> P, pair<T, T> Q, pair<T, T> O) {
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

Parâmetros:

* `(P, Q)`: Pontos extremos do segmento `PQ`

Retorna: Reta mediatriz ao segmento

```c++
template <typename T>
Line<T> perpendicularBisector(const pair<T, T>& P, const pair<T, T>& Q) {
    T a = 2 * (Q.x - P.x), b = 2 * (Q.y - P.y);
    T c = (P.x * P.x + P.y * P.y) - (Q.x * Q.x + Q.y * Q.y);
    return { a, b, c };
}
```

### Rotação de ponto

Parâmetros:

* `(P)`: Ponto alvo
* `(radians)`: Ângulo em radianos

Retorna: Ponto rotacionado

```c++
template <typename T>
pd rotate(const pair<T, T>& P, double radians) {
    double x = cos(radians) * P.x - sin(radians) * P.y;
    double y = sin(radians) * P.x + cos(radians) * P.y;
    return { x, y };
}
```

## Árvores

### Binary lifting

Parâmetros:

* `(g)`: Árvore/Grafo alvo
* `(n)`: Quantidade de nós
* `(u)`: Nó alvo
* `(k)`: Ordem do ancestral

Retorna: `k` ancestral do nó `u`

```c++
const ll LOG = 31; // ceil log of n

vvll parent;
vll depth;

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

ll kthAncestor(ll u, ll k) {
    if (k > depth[u]) return -1; // no kth ancestor
    rep(i, 0, LOG)
        if (k & (1 << i))
            u = parent[u][i];
    return u;
}
```

### Centróide

Parâmetros:

* `(g)`: Árvore alvo
* `(u)`: Raiz da árvore

Retorna: Nova raiz na qual nenhuma outra subárvore tem mais que `n / 2` nós

```c++
vll subtree;

ll dfs(const vvll& g, ll u, ll p = 0) {
    subtree[u] = 1;
    for (ll v : g[u]) if (v != p)
        subtree[u] += dfs(g, v, u);
    return subtree[u];
}

ll centroid(const vvll& g, ll u, ll p = 0) {
    for (ll v : g[u]) if (v != p)
        if (subtree[v] * 2 > g.size() - 1)
            return centroid(g, v, u);
    return u;
}
```

### Euler Tour

Parâmetros:

`(g)`: Árvore alvo
`(u)`: Raíz da árvore

Adendos:

* Técnica para linearizar árvores, popularemos um vetor de início e fim para cada nó
que marca o intervalo que representa a subárvore de cada nó. Assim, podemos fazer
operações nesses intervalos

```c++
ll timer = 0;
vll st, et;
void eulerTour(const vvll& g, ll u, ll p = 0) {
    if (st.empty())
        st.resize(g.size()), et.resize(g.size());
    st[u] = timer++;
    for (ll v : g[u]) if (v != p)
        eulerTour(g, v, u);
    et[u] = timer++;
}
```

### Menor ancestral comum (LCA)

Parâmetros:

* `(u, v)`: Nós alvos

Retorna: Menor ancestral comum entre os nós

Adendos:

* Necessário a técnica de binary lifting

```c++
ll lca(ll u, ll v) {
    if (depth[u] < depth[v]) swap(u, v);
    ll k = depth[u] - depth[v];
    u = kthAncestor(u, k);
    if (u == v) return u;
    per(i, LOG - 1, 0)
        if (parent[u][i] != parent[v][i])
            u = parent[u][i], v = parent[v][i];
    return parent[u][0];
}
```

## Grafos

### Bellman-Ford

Parâmetros:

`(g)`: Grafo alvo
`(s): Vértice inicial (menores distâncias em relação à ele)

Retorna: Vetor com as menores distâncias de cada vértice para `s` e vetor de trajetos

```c++
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
                cnt[v]++;
                pre[v] = u;
                if (cnt[v] == g.size()) {
                    ds[v] = LLONG_MIN;
                    ds[0] = v; // a node that has -inf dist
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

Parâmetros:

* `(g)`: Grafo alvo
* `(s)`: Vértice inicial (menores distâncias em relação à ele)

Retorna: Vetor com as menores distâncias de cada vértice para `s`

```c++
vll bfs01(const vvpll& g, ll s) {
    vll ds(g.size(), LLONG_MAX);
    deque<ll> dq;
    dq.emplace(s); ds[s] = 0;
    while (!dq.empty()) {
        ll u = pq.front(); pq.pop();
        for (ll v : g[u])
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

* `(g)`: Grafo alvo
* `(directed)`: Flag para indicar se é um grafo direcionado ou não
* `(s)`: Vértice inicial
* `(e)`: Vértice final (Apenas em caminho euleriano)

Retorna: Vetor com o ciclo euleriano se só especificado `s`, se não caminho euleriano,
vazio se impossível ou se não houverem arestas
  
Adendos:

* Se só o vértice inicial for especificado procuraremos um ciclo euleriano
* Testei com os problemas `Mail Delivery`, `De Bruijin Sequence` e `Teleporters Path`,
  talvez seja necessário adicionar algumas checagens para ver se o caminho existe não
  olhei bem ainda
  
```c++
vll eulerianPath(const vvll& g, bool directed, ll s, ll e = -1) {
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
        return {};
        
    rep(u, 0, h.size()) {
        if (e != -1 and (u == s or u == e)) continue;
        if (in_degree[u] != h[u].size() or (!directed and in_degree[u] & 1))
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
            if (!directed) {
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

Parâmetros:

* `(g)`: Grafo alvo
* `(s)`: Vértice inicial (menores distâncias em relação à ele)

Retorna: Vetor com as menores distâncias de cada vértice para `s` e vetor de trajetos

Adendos:

* Perceba que quando pomos um vértice na fila não necessariamente será com o último
  tempo que ele vai ter, então se estivermos modificando para conseguir, por exemplo,
  a quantidade de menores caminhos ou a menor quantidade de vértices no menor caminho,
  se a nova distância for menor vai ser necessário resetar qualquer resposta calculada
  anteriormente

```c++
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
    } while (u != s);
    reverse(all(p));
    return p;
}
```

### Floyd Warshall

Parâmetros:

* `(g)`: Grafo alvo

Retorna: Vetor com as menores distâncias entre cada par de vértices

```c++
vvll floydWarshall(const vvpll& g) {
    ll n = g.size();
    vvll ds(n, vll(n, INT_MAX));

    rep(u, 1, n)
        ds[u][u] = 0;
        for (auto [w, v] : g[u]) {
            ds[u][v] = min(ds[u][v], w);
            if (ds[u][u] < 0)
                ds[u][u] = INT_MIN; // negative cycle
        }
    }

    rep(k, 1, n)
        rep(u, 1, n)
            rep(v, 1, n)
                if (ds[u][k] != INT_MAX and ds[k][v] != INT_MAX) {
                    if (ds[k][k] == INT_MIN)
                        ds[u][v] = INT_MIN;
                    else {
                        ds[u][v] = min(ds[u][v], ds[u][k] + ds[k][v]);
                        if (ds[u][u] < 0)
                            ds[u][u] = INT_MIN;
                    }
                }

    return ds;
}
```

### Kosaraju

Parâmetros:

`(g)`: Grafo alvo

Retorna: Par com o grafo condensado e as componentes fortemente conectadas

```c++
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

Parâmetros:

* `(edges)`: Vetor de arestas `(peso, u, v)`
* `(n)`: Quantidade máxima de vértices

Retorna: Vetor com a árvore geradora mínima (se o grafo for conectado), representado
por vetor de arestas e a soma total de suas arestas

```c++
pair<vtll, ll> kruskal(vtll& edges, ll n) {
    DSU dsu(n);
    vtll mst;
    ll edges_sum = 0;
    sort(all(edges));
    for (auto [w, u, v] : edges) if (!dsu.sameSet(u, v)) {
        dsu.mergeSetsOf(u, v);
        mst.eb(w, u, v);
        edges_sum += w;
    }
    return { mst, edges_sum };
}
```

### Ordenação topológica

Parâmetros:

* `(g)`: Grafo alvo

Retorna: Vetor com a ordenação topológica do grafo (vazio se houver ciclo)

```c++
vll topologicalSort(vvll& g) {
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

    if (res.size() != g.size() - 1) return {}; // cycle
    return res;
}
```

## Outros

### Maior subsequência comum (LCS)

Parâmetros:

* `(xs, ys)`: Vetores alvo

Retorna: Tamanho da maior subsequência comum

```c++
template <typename T>
ll lcs(T& xs, T& ys) {
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

Parâmetros:

* `(xs)`: Vetor alvo

Retorna: Tamanho da maior subsequência crescente e último elemento dela

```c++
pll lis(vll& xs) {
    vll ss;
    for (ll x : xs) {
        auto it = lower_bound(all(ss), x);
        if (it == ss.end()) ss.eb(x);
        else *it = x;
    }
    return { ss.size(), ss.back() };
}
```

### Soma de prefixo 2D

```c++
struct Psum2D {
    Psum2D(const vvll& xs) : n(xs.size()), m(xs[0].size()), psum(n + 1, vll(m + 1)) {
        rep(i, 0, n)
            rep(j, 0, m) {
                // sum side and up rectangles, add element and remove intersection
                psum[i + 1][j + 1] = psum[i + 1][j] + psum[i][j + 1];
                psum[i + 1][j + 1] += xs[i][j] - psum[i][j];
            }
    }

    ll query(ll lx, ll hx, ll ly, ll hy) {
        // sum total rectangle, subtract side and up and add intersection
        ll res = psum[hy][hx] - psum[hy][lx - 1] - psum[ly - 1][hx];
        res += psum[ly - 1][lx - 1];
        return res;
    }

    ll n, m;
    vvll psum;
}
```

### Soma de prefixo 3D

```c++
struct Psum3D {
    Psum3D(const vvvll& xs)
            : n(xs.size()), m(xs[0].size()), o(xs[0][0].size()),
              psum(n + 1, vvll(m + 1, vll(o + 1)) {
        rep(i, 1, n + 1)
            rep(j, 1, m + 1)
                rep(k, 1, o + 1) {
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

    ll query(ll lx, ll hx, ll ly, ll hy, ll lz, ll hz) {
        // sum total cuboid, subtract sides and down
        ll res = psum[hx][hy][hz]     - psum[lx - 1][hy][hz] -
                 psum[hx][ly - 1][hz] - psum[hx][hy][lz - 1];
        // add intersections
        res += psum[hx][ly - 1][lz - 1] + psum[lx - 1][hy][lz - 1] +
                                          psum[lx - 1][ly - 1][hz];
        // re-subtract missing cuboid
        res -= psum[lx - 1][ly - 1][lz - 1];
        return res;
    }

    ll n, m, o;
    vvvll psum;
}
```

## Matemática

### Coeficiente binomial

Retorna: Combinação de `n` elementos tomados `p` módulo `M`

```c++
ll binom(ll n, ll p) {
    const ll MAXN = 5e5, M = 1e9 + 7; // check mod value!
    static ll fac[MAXN + 1], inv[MAXN + 1], finv[MAXN + 1];
    if (fac[0] != 1) {
        fac[0] = fac[1] = inv[1] = finv[0] = finv[1] = 1;
        rep(i, 2, MAXN + 1) {
            fac[i] = fac[i - 1] * i % M;
            inv[i] = M - M / i * inv[M % i] % M;
            finv[i] = finv[i - 1] * inv[i] % M;
        }
    }
    if (n < p or n * p < 0) return 0;
    return fac[n] * finv[p] % M * finv[n - p] % M;
}
```

### Conversão de base

Parâmetros:

* `(x)`: Número alvo
* `(b)`: Base desejada

Retorna: Vetor com a representação de `x` na base `b`

```c++
// coefficients like (b = 2) 1*b^2 + 0*b^1 + 1*b^0 = 5
vll toBase(ll x, ll b) {
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

Parâmetros:

* `(n)`: Tamanho do intervalo desejado

Retorna: Vetor com todos os primos no intervalo `[1, n]` e vetor de menor fator primo

```c++
pair<vll, vll> sieve(ll n) {
    vll ps, spf(n + 1);
    spf[1] = 1;
    rep(i, 2, n + 1) if (!spf[i]) {
        ps.eb(i);
        for (ll j = i; j <= n; j += i)
            if (!spf[j]) spf[j] = i;
    }
    return { ps, spf };
}
```

Adendos:

* Exemplo de fatoração em `O(logN)` com o vetor de menor fator primo:

```c++
vll factors(ll x, const vll& spf) {
    vll fs;
    while (x != 1) {
        fs.eb(spf[x]);
        x /= spf[x];
    }
    return fs;
}
```

### Divisores

Parâmetros:

* `(x)`: Número alvo

Retorna: Vetor desordenado com todos os divisores

```c++
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

### Fatoração

Parâmetros:

* `(x)`: Número alvo

Retorna: Vetor com os fatores primos

```c++
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

### GCD

```c++
ll gcd(ll a, ll b) {
    while (b) {
        a %= b;
        swap(a, b);
    }
    return a;
}
```

### LCM

```c++
ll lcm(ll a, ll b) {
    return a / gcd(a, b) * b;
}
```

### Quantidade de divisores

Parâmetros:

* `(x)`: Número alvo

Adendos:

* Primeira chamada realiza processamento `O(NlogN)`, depois retorna em `O(1)`

```c++
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

## Strings

### Z-Function

```c++
vll z(const string &s) {
    ll n = s.size(), l = 0, r = 0;
    vll zs(n);

    rep(i, 1, n) {
        if (i <= r)
            zs[i] = min(zs[i - l], r - i + 1);

        while (zs[i] + i < n && s[zs[i]] == s[i + zs[i]])
            zs[i]++;

        if (r < i + zs[i] - 1)
            l = i, r = i + zs[i] - 1;
    }

    return zs;
}
```

# Estruturas

## Árvores

### BIT tree 2D

Parâmetros:

* `(h, w)`: Dimensões que definem os intervalos máximos para operações `[1, h]` e `[1, w]`

Métodos:

* `add(y, x, v)`: Adiciona `v` na posição `(y, x)`
* `sum(ly, lx, hy, hx)`: Retorna a soma do retângulo demarcado pelos pontos `(ly, lx)` e `(hy, hx)`

```c++
template <typename T>
struct BIT2D {
    BIT2D(ll h, ll w) : n(h), m(w), bit(n + 1, vector<T>(m + 1)) {}
    
    void add(ll y, ll x, T v) {
       	for (; y <= n; y += y & -y)
            for (ll i = x; i <= m; i += i & -i)
                bit[y][i] += v;
    }
    
    T sum(ll y, ll x) {
        T sum = 0;
        for (; y > 0; y -= y & -y)
            for (ll i = x; i > 0; i -= i & -i)
                sum += bit[y][i];
        return sum;
    }
    
    T sum(ll ly, ll lx, ll hy, ll hx) {
        return sum(hy, hx)     - sum(hy, lx - 1) -
               sum(ly - 1, hx) + sum(ly - 1, lx - 1);
    }
    
    ll n, m;
    vector<vector<T>> bit;
};
```

### Disjoint set union

Parâmetros:

* `(n)`: Intervalo máximo para operações `[0, n]`

Métodos:

* `setOf(x)`: Retorna o representante do conjunto que contém `x`
* `mergeSetsOf(x, y)`: Junta os conjuntos que contém `x` e `y`
* `sameSet(x, y)`: Retorna se `x` e `y` estão contidos no mesmo conjunto

```c++
struct DSU {
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

    vll parent, size;
};
```

### Red-Black tree (ordered set)

Métodos:

* `find_by_order(k)`: Retorna o elemento no índice `k`
* `order_of_key(x)`:  Retorna quantos elementos existem estritamente menores que `x`

```c++
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace __gnu_pbds;

template <typename T>
struct RBT {
    void insert(T x) { rb.insert({ x, n++ }); }
    ll order_of_key(T x) { return rb.order_of_key({ x, 0 }); }
    ll size() { return rb.size(); };
    bool empty() { return rb.empty(); };
        
    void erase(T x) {
        auto it = rb.lower_bound({ x, 0 });
        if (it == rb.end() or it->first != x) return;
        rb.erase(it);
    }
    
    T find_by_order(ll k) { 
        auto it = rb.find_by_order(k);
        if (it == rb.end()) return -1; // value for not found
        return it->first;
    }

    ll n = 1;
    tree<pair<T, ll>, null_type, less<>,
    rb_tree_tag, tree_order_statistics_node_update> rb;
};
```

```c++
template <typename T> // unique elements (set)
using RBT = tree<T, null_type, less<>,
rb_tree_tag, tree_order_statistics_node_update>;
```

### Segment tree

Parâmetros:

* `(sz)`: Intervalo máximo para operações `[0, sz)`
* `(f)`: Função desejada `(max, min, gcd, ...)`
* `(def)`: Valor padrão (`LLONG_MIN` se `op = max`, `LLONG_MAX` se `op = min`, ...)

Métodos:

* `setQuery(i, j, x)`: super função (set/query). intervalo da
  ação `[i, j]`. `x` é um argumento opcional que é passado
  somente quando queremos setar valores. retorna valor
  correspondente à funcionalidade codificada

Adendos:

* Dependendo da operação é necessário fazer ajustes

```c++
template <typename T, typename Op = function<T(T, T)>>
struct Segtree {
    Segtree(ll sz, Op f, T def)
        : seg(4 * sz, def), lzy(4 * sz), n(sz), op(f), DEF(def) {}

    T setQuery(ll i, ll j, ll x = LLONG_MIN, ll l = 0, ll r = -1, ll no = 1) {
        if (r == -1) r = n - 1;
        if (lzy[no]) unlazy(no, l, r);
        if (j < l or i > r) return DEF;
        if (i <= l and r <= j) {
            if (x != LLONG_MIN) { 
                lzy[no] += x;
                unlazy(no, l, r);
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
    void unlazy(ll no, ll l, ll r) {
        if (seg[no] == DEF) seg[no] = 0;
        seg[no] += (r - l + 1) * lzy[no]; // sum
        // seg[no] += lzy[no]; // min/max
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

### Wavelet Tree

Parâmetros:

* `(xs)`: Vetor comprimido alvo
* `(n)`: Quantidade de elementos distintos em `xs`

Métodos:

* `kTh(l, r, k)`: retorna o `k`-ésimo (`k > 0`) menor elemento no intervalo `[l, r]`

Adendos:

* 0-indexed
* Ordena o vetor `xs` no processo de construção
* Capaz de retornar as ocorrências de um elemento em um intervalo, mas dá para fazer isso
  trivialmente com uma matriz com as posições dos elementos e uma busca binária

```c++
struct WaveletTree {
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

Parâmetros:

* `(P, Q)`: Pontos distintos contidos na reta

Métodos:

* `contains(P)`: Retorna se `P` está contido na reta
* `parallel(r)`: Retorna se a reta é paralela a `r`
* `orthogonal(r)`: Retorna se a reta é perpendicular a `r`
* `intersection(r)`: Retorna ponto de interseção com `r`
* `distance(P)`: Retorna a distância de `P` à reta
* `closest(P)`: Retorna ponto na reta mais próximo de `P`

```c++
template <typename T>
struct Line {
    Line(const pair<T, T>& P, const pair<T, T>& Q)
            : a(P.y - Q.y), b(Q.x - P.x), c(P.x * Q.y - Q.x * P.y) {
        if constexpr (is_floating_point_v<T>) b /= a, c /= a, a = 1;
        else {
            if (a < 0 or (a == 0 and b < 0)) a *= -1, b *= -1, c *= -1;
            T gcd_abc = gcd(a, gcd(b, c));
            a /= gcd_abc, b /= gcd_abc, c /= gcd_abc;
        }
    }

    bool contains(const pair<T, T>& P) { return equals(a * P.x + b * P.y + c, 0); }

    bool parallel(const Line& r) {
        T det = a * r.b - b * r.a;
        return equals(det, 0);
    }

    bool orthogonal(const Line& r) { return equals(a * r.a + b * r.b, 0); }

    pd intersection(const Line& r) {
        double det = r.a * b - r.b * a;

        // same or parallel
        if (equals(det, 0)) return {};

        double x = (-r.c * b + c * r.b) / det;
        double y = (-c * r.a + r.c * a) / det;
        return { x, y };
    }

    double distance(const pair<T, T>& P) {
        return abs(a * P.x + b * P.y + c) / hypot(a, b);
    }

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

Parâmetros:

* `(P, Q)`: Pontos extremos do segmento

Métodos:

* `contains(P)`: Retorna se `P` está contido no segmento
* `intersect(r)`: Retorna se `r` intersecta com o segmento
* `closest(P)`: Retorna ponto mais próximo no segmento de `P`

```c++
template <typename T>
struct Segment {
    Segment(const pair<T, T>& P, const pair<T, T>& Q) : A(P), B(Q) {}

    bool contains(const pair<T, T>& P) const {
        T xmin = min(A.x, B.x), xmax = max(A.x, B.x);
        T ymin = min(A.y, B.y), ymax = max(A.y, B.y);
        if (P.x < xmin || P.x > xmax || P.y < ymin || P.y > ymax) return false;
        return equals((P.y - A.y) * (B.x - A.x), (P.x - A.x) * (B.y - A.y));
    }

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

    // closest point in segment to P
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

Parâmetros:

* `(P)`: Ponto do centro
* `(r)`: Raio

Métodos:

* `area()`: Retorna área
* `perimeter()`: Retorna perímetro
* `arc(radians)`: Retorna comprimento do arco
* `chord(radians)`: Retorna comprimento da corda
* `sector(radians)`: Retorna área do setor
* `segment(radians)`: Retorna área do segmento
* `position(P)`: Retorna valor que representa a posição do ponto
* `intersection(c)`: Retorna pontos de interseção com círculo
* `intersection(P, Q)`: Retorna pontos de interseção com reta `PQ`
* `tanPoints()`: Retorna dois pontos tangentes ao círculo à partir da origem
* `from3(P, Q, R)`: Retorna o círculo formado por esses pontos
* `mec(ps)`: Retorna o menor círculo que contém todos os pontos de `ps`

```c++
enum Position { IN, ON, OUT };

template <typename T>
struct Circle {
    Circle(const pair<T, T>& P, T r) : C(P), r(r) {}

    double area() { return acos(-1.0) * r * r; }
    double perimeter() { return 2.0 * acos(-1.0) * r; }
    double arc(double radians) { return radians * r; }
    double chord(double radians) { return 2.0 * r * sin(radians / 2.0); }
    double sector(double radians) { return (radians * r * r) / 2.0; }

    double segment(double radians) {
        double c = chord(radians);
        double s = (r + r + c) / 2.0;
        double t = sqrt(s) * sqrt(s - r) * sqrt(s - r) * sqrt(s - c);
        return sector(radians) - t;
    }

    Position position(const pair<T, T>& P) {
        double d = dist(P, C);
        return equals(d, r) ? ON : (d < r ? IN : OUT);
    }

    vector<pair<T, T>> intersection(const Circle& c) {
        double d = dist(c.C, C);

        // no intersection or same
        if (d > c.r + r or d < abs(c.r - r) or (equals(d, 0) and equals(c.r, r)))
            return {};

        double a = (c.r * c.r - r * r + d * d) / (2.0 * d);
        double h = sqrt(c.r * c.r - a * a);
        double x = c.C.x + (a / d) * (C.x - c.C.x);
        double y = c.C.y + (a / d) * (C.y - c.C.y);
        pd P1, P2;
        P1.x = x + (h / d) * (C.y - c.C.y);
        P1.y = y - (h / d) * (C.x - c.C.x);
        P2.x = x - (h / d) * (C.y - c.C.y);
        P2.y = y + (h / d) * (C.x - c.C.x);
        return P1 == P2 ? vector<pair<T, T>> { P1 } : vector<pair<T, T>> { P1, P2 };
    }

    // circle at origin
    vector<pd> intersection(const pair<T, T>& P, const pair<T, T>& Q) {
        double a(P.y - Q.y), b(Q.x - P.x), c(P.x * Q.y - Q.x * P.y);
        double x0 = -a * c / (a * a + b * b), y0 = -b * c / (a * a + b * b);
        if (c*c > r*r * (a*a + b*b) + 1e-9) return {};
        if (equals(c*c, r*r * (a*a + b*b))) return { { x0, y0 } };
        double d = r * r - c * c / (a * a + b * b);
        double mult = sqrt(d / (a * a + b * b));
        double ax = x0 + b * mult;
        double bx = x0 - b * mult;
        double ay = y0 - a * mult;
        double by = y0 + a * mult;
        return { { ax, ay }, { bx, by } };
    }

    // from origin
    pair<pd, pd> tanPoints() {
        double b = hypot(C.x, C.y), th = acos(r / b);
        double d = atan2(-C.y, -C.x), d1 = d + th, d2 = d - th;
        return { {C.x + r * cos(d1), C.y + r * sin(d1)},
                 {C.x + r * cos(d2), C.y + r * sin(d2)} };
    }

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

    static Circle<double> mec(vector<pair<T, T>>& ps) {
        random_shuffle(all(ps));
        Circle<double> c(ps[0], 0);
        rep(i, 0, ps.size()) {
            if (c.position(ps[i]) != OUT) continue;
            c = { ps[i], 0 };
            rep(j, 0, i) {
                if (c.position(ps[j]) != OUT) continue;
                c = {
                    { (ps[i].x + ps[j].x) / 2.0, (ps[i].y + ps[j].y) / 2.0 },
                       dist(ps[i], ps[j]) / 2.0
                };
                rep(k, 0, j)
                    if (c.position(ps[k]) == OUT)
                        c = from3(ps[i], ps[j], ps[k]);
            }
        }
        return c;
    }

    pair<T, T> C;
    T r;
};
```

### Triângulo

Parâmetros:

* `(P, Q, R)`: Pontos extremos do triângulo

Métodos:

* `area()`: Retorna área
* `perimeter()`: Retorna perímetro
* `sidesClassification()`: Retorna valor que representa a classificação do triângulo
* `anglesClassification()`: Retorna valor que representa a classificação do triângulo
* `barycenter()`: Retorna ponto de interseção entre as medianas
* `circumradius()`: Retorna valor do raio da circunferência circunscrita
* `circumcenter()`: Retorna ponto de interseção entre as as retas perpendiculares que
                    interceptam nos pontos médios
* `inradius()`: Retorna valor do raio da circunferência inscrita
* `incenter()`: Retorna ponto de interseção entre as bissetrizes
* `orthocenter()`: Retorna ponto de interseção entre as alturas

```c++
enum Class { EQUILATERAL, ISOSCELES, SCALENE };
enum Angles { RIGHT, ACUTE, OBTUSE };

template <typename T>
struct Triangle {
    Triangle(pair<T, T> P, pair<T, T> Q, pair<T, T> R)
        : A(P), B(Q), C(R), a(dist(A, B)), b(dist(B, C)), c(dist(C, A)) {}

    T area() {
        T det = (A.x * B.y + A.y * C.x + B.x * C.y) -
                (C.x * B.y + C.y * A.x + B.x * A.y);
        if (is_floating_point_v<T>) return 0.5 * abs(det);
        return abs(det);
    }

    double perimeter() { return a + b + c; }

    Class sidesClassification() {
        if (equals(a, b) and equals(b, c)) return EQUILATERAL;
        if (equals(a, b) or equals(a, c) or equals(b, c)) return ISOSCELES;
        return SCALENE;
    }

    Angles anglesClassification() {
        double alpha = acos((a * a - b * b - c * c) / (-2.0 * b * c));
        double beta = acos((b * b - a * a - c * c) / (-2.0 * a * c));
        double gamma = acos((c * c - a * a - b * b) / (-2.0 * a * b));
        double right = acos(-1.0) / 2.0;
        if (equals(alpha, right) || equals(beta, right) || equals(gamma, right))
            return RIGHT;
        if (alpha > right || beta > right || gamma > right) return OBTUSE;
        return ACUTE;
    }

    pd barycenter() {
        double x = (A.x + B.x + C.x) / 3.0;
        double y = (A.y + B.y + C.y) / 3.0;
        return {x, y};
    }

    double circumradius() { return (a * b * c) / (4.0 * area()); }

    pd circumcenter() {
        double D = 2 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y));
        T A2 = A.x * A.x + A.y * A.y, B2 = B.x * B.x + B.y * B.y,
                                      C2 = C.x * C.x + C.y * C.y;
        double x = (A2 * (B.y - C.y) + B2 * (C.y - A.y) + C2 * (A.y - B.y)) / D;
        double y = (A2 * (C.x - B.x) + B2 * (A.x - C.x) + C2 * (B.x - A.x)) / D;
        return {x, y};
    }

    double inradius() { return (2 * area()) / perimeter(); }

    pd incenter() {
        double P = perimeter();
        double x = (a * A.x + b * B.x + c * C.x) / P;
        double y = (a * A.y + b * B.y + c * C.y) / P;
        return {x, y};
    }

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

Parâmetros:

* `(ps)`: Vetor de pontos (representando polígono, sem repetir ponto inicial)

Métodos:

* `convex()`: Retorna se o polígono é convexo
* `area()`: Retorna área
* `perimeter()`: Retorna perímetro
* `contains(P)`: Retorna se o polígono contém o ponto `P`
* `cut(P, Q)`: Retorna polígono separado pela reta `PQ`
* `circumradius()`: Retorna valor do raio da circunferência circunscrita
* `apothem()`: Retorna valor da apótema

```c++
template <typename T>
struct Polygon {
    // should be clock ordered
    Polygon(const vector<pair<T, T>>& ps)
        : vs(ps), n(vs.size()) { vs.eb(vs.front()); }

    bool convex() {
        if (n < 3) return false;
        ll P = 0, N = 0, Z = 0;

        rep(i, 0, n) {
            auto d = D(vs[i], vs[(i + 1) % n], vs[(i + 2) % n]);
            d ? (d > 0 ? ++P : ++N) : ++Z;
        }

        return P == n or N == n;
    }

    T area() { // double if lattice
        T a = 0;

        rep(i, 0, n)
            a += vs[i].x * vs[i + 1].y - vs[i + 1].x * vs[i].y;

        if (is_floating_point_v<T>) return 0.5 * abs(a);
        return abs(a);
    }

    double perimeter() {
        double p = 0;
        rep(i, 0, n)
            p += dist(vs[i], vs[i + 1]);
        return p;
    }

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

        return equals(abs(sum), 2 * acos(-1.0)); // check precision
    }

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

    // for regular polygons
    double circumradius() {
        double s = dist(vs[0], vs[1]);
        return (s / 2.0) * (1.0 / sin(acos(-1.0) / n));
    }

    // for regular polygons
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
        rep(i, 0, n)
            rep(j, 0, n)
                rep(k, 0, n)
                    res[i][k] += mat[i][j] * other[j][k];
        return res;
    }

    static Matrix pow(const Matrix& matrix, ll exp) {
        ll n = matrix.n;
        Matrix tmp = matrix, res(n);
        rep(i, 0, n) res[i][i] = 1;
        while (exp > 0) {
            if (exp & 1) res = res * tmp;
            tmp = tmp * tmp;
            exp /= 2;
        }
        return res;
    }

    vector<vector<T>> mat;
    ll n;
};
```

## Strings

### Hash

Parâmetros:

* `(s)`: String alvo

Métodos:

* `get(i, j)`: Retorna o hash da substring `[i, j]` em O(1)

```c++
struct Hash {
    static const ll M1 = 1e9 + 7, M2 = 1e9 + 9, p1 = 31, p2 = 29;
    #define T pair<Mi<M1>, Mi<M2>>
    
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
# Utils

### Aritmética modular

```c++
const ll MOD = 1e9 + 7;
template <ll M = MOD>
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
    Mi operator/=(Mi b) & { return *this *= pow(b, M - 2); }
    friend Mi operator+(Mi a, Mi b) { return a += b; }
    friend Mi operator-(Mi a, Mi b) { return a -= b; }
    friend Mi operator*(Mi a, Mi b) { return a *= b; }
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
struct Bi {
    string v;
    Bi() : v("0") {}
    Bi(const string& x) : v(x) { reverse(v.begin(), v.end()); }
    friend Bi operator+(Bi a, const Bi& b) { return a += b; }
    friend Bi operator-(Bi a, const Bi& b) { return a -= b; }

    friend ostream& operator<<(ostream& os, const Bi& a) {
        ll i = a.v.size() - 1;
        while (a.v[i] == '0' and i > 0) --i;
        while (i >= 0) os << a.v[i--];
        return os;
    }

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

    // assumes a > b
    Bi operator-=(const Bi& b) {
        rep(i, 0, v.size()) {
            ll x = v[i] - '0';
            if (i < b.v.size()) x -= b.v[i] - '0';
            if (x < 0) x += 10, --v[i + 1];
            v[i] = x + '0';
        }
        return *this;
    }

    Bi prefix(ll n) {
        string p = v.substr(v.size() - n, n);
        reverse(all(p));
        return p;
    }

    Bi suffix(ll n) {
        string s = v.substr(0, n);
        reverse(all(s));
        return s;
    }
};
```

### Ceil division

```c++
ll ceilDiv(ll a, ll b) { return a / b + ((a ^ b) > 0 && a % b != 0); }
```

### Conversão de índices

```c++
#define K(i, j) ((i) * w + (j))
#define I(k)    ((k) / w)
#define J(k)    ((k) % w)
```

### Compressão de coordenadas

```c++
map<ll, ll> compress(vll& xs) {
    ll c = 0;
    set<ll> ys(all(xs));
    map<ll, ll> mp;
    for (ll y : ys)
        mp[y] = c++;
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

> Maior quantidade de primos na fatoração de um número `< 10^6` é `19`

> Maior quantidade de primos na fatoração de um número `< 10^3` é `9`

> Números primos interessantes: `2^31 - 1, 2^31 + 11, 10^18 - 11, 10^18 + 3`.

> `gcd(a, b) = gcd(a, a - b)`, `gcd(a, b, c) = gcd(a, a - b, a - c)`

> Divisibilidade por `3`: Soma dos algarismos divisível por `3`

> Divisibilidade por `4`: Número formado pelos dois últimos algarismos divísivel por `4`

> Divisibilidade por `6`: Par e divísivel por `3`

> Divisibilidade por `7`: Dobro do último algarismo subtraído do número sem ele divisível por `7` (pode ir repetindo)

> Divisibilidade por `8`: Número formado pelos três últimos algarismos divísivel por `8`

> Divisibilidade por `9`: Soma dos algarismos divisível por `9`

> Divisibilidade por `11`: Diferença entre soma dos algarismos de ordem ímpar e par divisível por `11`

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

### Igualdade flutuante

```c++
template <typename T, typename S>
bool equals(T a, S b) { return abs(a - b) < 1e-9; }
```

### Intervalos com soma S

```c++
// can change to count odd/even intervals
ll countIntervals(vll& xs, ll sum) {
    map<ll, ll> freq;
    ll ans = 0, psum = 0;
    freq[0] = 1;
    for (ll x : xs) {
        psum += x;
        ans += freq[psum - sum];
        ++freq[psum];
    }
    return ans;
}
```

### Kadane

```c++
ll kadane(const vll& xs) {
    ll res = 0, csum = 0;
    rep(i, 0, xs.size()) {
        csum += xs[i];
        if (csum < 0) csum = 0;
        res = max(res, csum);
    }
    return res;
}
```

### Próximo maior/menor elemento

```c++
// get vector with indexes of closest biggest
vll closests(const vll& xs) {
    vll closest(xs.size(), -1);
    stack<ll> prevs;
    // 0 .. n: closest to the left
    rep(i, 0, xs.size()) { //                     <= if want smallest
        while (!prevs.empty() and xs[prevs.top()] <= xs[i])
            prevs.pop();
        if (!prevs.empty()) closest[i] = prevs.top();
        prevs.emplace(i);
    }
    return closest;
}
```

### Soma de todos os intervalos

```c++
// by counting in how many intervals an element appear
ll sumAllIntervals(const vll& xs) {
    ll sum = 0, opens = 0;
    rep(i, 0, xs.size()) {
        opens += xs.size() - 2 * i;
        sum += xs[i] * opens;
    }
    return sum;
}
```

```c++
// by adding each prefix sum
ll sumAllIntervals(const vll& xs) {
    ll sum = 0, csum = 0 ll n = xs.size();
    rep(i, 0, n)
        csum += xs[i] * (n - i);
    rep(i, 0, n) {
        sum += csum;
        csum -= xs[i] * (n - i);
    }
    return sum;
}
```
