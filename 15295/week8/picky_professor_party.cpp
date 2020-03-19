// Maximum flow using Ford-Fulkerson
//
// Author      : Daniel Anderson
// Date        : 28-08-2016
//
// Usage:
//    MaxFlow G(n)
//      Create a flow network G with n nodes
//
//    G.max_flow(int s, int t)
//      Returns the maximum flow s -> t
//
//	  G.add_edge(u, v, cap)
//	    Adds an edge from u -> v with capacity. Returns the index
//      of the edge.
//
//    G.get_edge(i)
//      Returns a reference to the i'th edge. Use to check flows
//      or update capacities
//
//  Time Complexity: O(Ef), where f is the maximum flow and E
//    is the number of edges in the network.

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef vector<vi> vvi;

const ll INF = numeric_limits<ll>::max();

class MaxFlow {
  struct edge {
    int to;
    ll flow, cap;
  };
  int n;
  vector<edge> edges;
  vvi g;
  vi vis;

  ll dfs(int u, int t, ll flow) {
    if (u == t)
      return flow;
    vis[u] = true;
    for (auto id : g[u]) {
      edge &e = edges[id];
      edge &rev = edges[id ^ 1];
      ll residual = e.cap - e.flow, augment = 0;
      if (vis[e.to] || residual <= 0)
        continue;
      if ((augment = dfs(e.to, t, min(flow, residual))) > 0) {
        e.flow += augment;
        rev.flow -= augment;
        return augment;
      }
    }
    return 0;
  }

public:
  // Initialise a flow network with n nodes
  MaxFlow(int n) : n(n), g(n) {}

  // Add an edge with capacity cap from node u to node v
  // Returns the index of the edge.
  int add_edge(int u, int v, ll cap) {
    g[u].push_back((int)edges.size());
    edges.push_back({v, 0, cap});
    g[v].push_back((int)edges.size());
    edges.push_back({u, 0, 0}); // Change to {u, 0, cap} for bidirectional edges
    return (int)edges.size() - 2;
  }

  // Get a reference to a specific edge: use to check flows or update capcities
  edge &get_edge(int i) { return edges[i]; }

  // Return the max flow from s to t
  ll max_flow(int s, int t) {
    for (auto &e : edges)
      e.flow = 0;
    ll flow = 0, augment = 0;
    while (vis.assign(n, 0), (augment = dfs(s, t, INF)) != 0) {
      flow += augment;
    }
    return flow;
  }
};

int main(int argc, char *argv[]) {
    int n, m;
    cin >> n >> m;
    //1 based problem, take care of index
    ll drink=0;
    MaxFlow G(n+m+2);
    vector<int> beer (m);
    vector<int> prof (n);
    // vector<int> edge_idx;
    for (int i=0; i<n; i++) {
        int num;
        cin >> num;
        // cout << num;
        prof[i]=num;
        int idx = G.add_edge(i+m+1,n+m+1,num);
        // cout << "Prof: " << i+m+1 << "End: " << n+m+1 << "Cap: "  << num << endl;
        drink += num;
        // edge_idx.push_back(idx);
    }
    // cout << endl;
    for (int j=0; j<m; j++) {
        int num; 
        cin >> num;
        // cout << num;
        beer[j]=num;
        // cout << "Start: " << 0 << "Beer: " << j+1 << "Cap: " << num << endl;
        G.add_edge(0, j+1, num);
    }
    // cout << endl;
    for (int i=0; i<n; i++) {
        int k;
        cin >> k;
        for (int j=0; j<k; j++) {
            int num;
            cin >> num;
            // cout << "Beer: " << num << "Prof: " << i+m+1 << "Cap: " << beer[j] << endl;
            G.add_edge(num, i+m+1, beer[j-1]);
        }
    }
    // cout << endl;
    ll flow = G.max_flow(0, m+n+1);
    
    // cout << flow << endl;
    // for (int c=0; c<edge_idx.size(); c++) {
    //     if (G.get_edge(c).cap != G.get_edge(c).flow) {
            
    //         cout << "NO" << endl;
    //         return 0;
    //     }
    // }
    
    if (drink == flow) {
        cout << "PARTY" << endl;
    }
    else
    {
        cout << "NO" << endl;
    }
}