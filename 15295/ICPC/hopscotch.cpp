#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <stdlib.h>

using namespace std;

typedef long long ll;
template<typename T>
using min_heap = priority_queue<T, vector<T>, greater<T>>;
const ll INF = 1e15;

vector<vector<pair<int,int>>> pos;
vector<vector<pair<ll,ll>>> adj;

ll cal_dist(const pair<int,int>&a, const pair<int,int>&b) {
    return abs(a.first-b.first)+abs(a.second-b.second);
}

int dijkstra(ll s, vector<ll> &dist) {
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
    // for (int i=0; i<dist.size(); i++) {
    //     cout << i << "  " << dist[i] << endl;
    // }
    return dist.back();
}

int main(int argc, char *argv[]) {
    int n, k;
    cin >> n >> k;
    vector<ll> dist(n*n+2,INF);
    pos.assign(k, vector<pair<int,int>> {});
    adj.assign(n*n+2, vector<pair<ll,ll>> {});
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            int cur;
            cin >> cur;
            pos[cur-1].push_back(make_pair(i,j));
        }
    }
    for (int i=0; i<pos.size(); i++) {
        if (pos[i].empty()) {
            cout << -1 << endl;
            return 0;
        }
    }
    // cout << pos.back().back().second << endl;
    int counter=1;
    for (int i=0; i<pos.front().size(); i++) {
        adj[0].push_back(make_pair(i,0));
    }
    for (int i=0; i+1<k; i++) {
        int l1=i,l2=i+1;
        for (int j=0; j<pos[l1].size(); j++) {
            // cout << counter+j << endl;
            for (int k=0; k<pos[l2].size(); k++) {
                ll dis=cal_dist(pos[l1][j],pos[l2][k]);
                adj[counter+j].push_back(make_pair(counter+pos[i].size()+k,dis));
            }
        }
        counter+=pos[i].size();
    }
    for (int i=0; i<pos.back().size(); i++) {
        adj[counter+i].push_back(make_pair(adj.size()-1,0));
    }
    adj[counter+pos.back().size()].push_back(make_pair(adj.size()-1,0));
    // for (int i=0; i<adj.size(); i++) {
    //     cout << adj[i].back().first << endl;
    // }
    int ans=dijkstra(0,dist);
    cout << ans << endl;
}

