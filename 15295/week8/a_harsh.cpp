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

int main()
{
    int p,b;
    ll p_total=0;
    cin>>p>>b;
    MaxFlow G(p+b+2);
    
    vi beer(b); // vector storing beer amouns available
    vi prof(p); // vector storing required beer amounts of id professor 
    
    
    
    
    
    //Adding professor to sink edges
    // for(int i=0 ; i<p ; i++)
    // {
    //     int b_temp; //professor beer amount
    //     cin>>b_temp;
    //     p_total+=b_temp;
    //     max_fl.add_edge(b+1+i,b+p+1,b_temp); //source node is 0
    // }

    for (int i=0; i<p; i++) {
        int num;
        cin >> num;
        // cout << num;
        p_total += num;
        prof[i]=num;
        int idx = G.add_edge(i+b+1,b+p+1,num);
        // cout << "Prof: " << i+m+1 << "End: " << n+m+1 << "Cap: "  << num << endl;
        
        // edge_idx.push_back(idx);
    }
    
    //Adding source to beer edges
    for(int i=0 ; i<b ; i++)
    {
        
        int b_amount; //professor beer amount
        cin>>b_amount;
        beer[i]=b_amount;
        // b_total+=b_amount;
        G.add_edge(0,i+1,b_amount); //source node is 0
    }
    
    
    //Adding beer to professor edges
    for(int i=0 ; i<p ; i++)
    {
        int num_beer_liked;
        cin >> num_beer_liked;
        for(int j=0 ; j<num_beer_liked ; j++)
        {
            int b_num; //beer number index
            cin>>b_num;
            
            G.add_edge(b_num,b+1+i,beer[b_num-1]); //source node is 0
        }
    }
    
    
    if(G.max_flow(0,p+b+1)==p_total)
      cout<<"PARTY";
      else cout<<"NO";
    
    
    
    return 0;
}