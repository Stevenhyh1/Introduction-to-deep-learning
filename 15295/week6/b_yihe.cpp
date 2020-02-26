#include <iostream>
#include <vector>
#include <queue>

using namespace std;

template<typename T>
using min_pq = priority_queue<T, vector<T>, greater<T>>;

typedef pair<long, long>  pi;

const long INF = 1e15;

int V, n, m;
vector<vector<pi>> adj;
vector<vector<int>> pos;

int dijkstra(int s, int t) {
    vector<long> dist(n*m, INF);
    dist[s] = 0;
    min_pq<pi> pq;
    pq.push({0,s});

    while (!pq.empty())
    {
        long u, d;
        d = pq.top().first;
        u = pq.top().second;
        pq.pop();

        if (d == dist[u]) {
            for (auto neighbour : adj[u]) {
                long v = neighbour.first, w = neighbour.second;
                if (dist[v] > dist[u]*)
            }
        }
    }
    
}

void build_adj(int x, int y, const vector<vector<int>> &pos) {
    int h=pos.size(), w=pos[0].size();
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            if (x+1<w) {
                adj[h*y+x].push_back({h*y+x+1, pos[y][x+1]});
                adj[h*y+x+1].push_back({h*y+x, pos[y][x+1]});
            }
            if (y+1<h) {
                adj[h*y+x].push_back({h*(y+1)+x, pos[y+1][x]});
                adj[h*(y+1)+x].push_back({h*y+x, pos[y+1][x]});
            }
        }
    }

}

int main(int argc, char *argv[]) {
    cin >> V >> n >> m;
    pos.assign(n, vector<int> (m, 0));
    adj.assign(m*n, vector<pi> {});
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            int val;
            cin >> val;
            pos[i][j] = val;
        }
    }
    build_adj(0, 0, pos);
    // cout << pos[2][1] << endl;
    for (int i=0; i<adj.size(); i++) {
        cout << adj[i].size() << endl;
    }
    
}
