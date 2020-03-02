#include<iostream>
#include<vector>
#include<string>
#include<queue>
#include<stdlib.h>

typedef long long ll;
using namespace std;

template<typename T>
using min_heap = priority_queue<T, vector<T>, greater<T>>;

const ll INF = 1e15;

ll n, m;
vector<vector<pair<ll,ll>>> adj;

void dijkstra(ll s, ll t, vector<ll> &dist) {
    min_heap<pair<ll,ll>> pq;
    pq.push({0,s});
    dist[s] = 0;

    while (!pq.empty()) {
        ll u,d;
        u = pq.top().second;
        d = pq.top().first;
        pq.pop();

        // cout << "vertex: " << u << endl;
        // cout << "distance from start: " << d << endl;

        if (d == dist[u]) {
            for (auto neighbour:adj[u]) {
                ll v=neighbour.first, w = neighbour.second;
                if (dist[v]>dist[u]+w) {
                    dist[v] = dist[u]+w;
                    pq.push({dist[v],v});
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    cin >> n >> m;
    vector<ll> dist_s(n,INF);
    vector<ll> dist_t(n,INF);
    adj.assign(n, vector<pair<ll,ll>>{});
    ll u,v,w;
    for (ll i=0; i<m; i++) {
        cin >> u >> v >> w;
        adj[u-1].push_back({v-1,w});
        adj[v-1].push_back({u-1,w});
    }
    // for (ll i=0; i<n; i++) {
    //     cout << adj[i].front().first << adj[i].front().second << endl;
    //     cout << adj[i].back().first << adj[i].back().second << endl;
    // }
    // cout << adj.size() << endl;
    dijkstra(0,n-1,dist_s);
    dijkstra(n-1,0,dist_t);
    ll ans=INF;
    for (ll j=0; j<n; j++) {
        if (ans>abs(dist_s[j]-dist_t[j])) {
            ans=abs(dist_s[j]-dist_t[j]);
        }
    }
    cout << ans << endl;
}