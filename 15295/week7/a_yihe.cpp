#include <iostream>
#include <queue>
#include <vector>
#include <algorithm>
#include <unordered_set>

using namespace std;

template<typename T>
using min_pq = priority_queue<T, vector<T>, greater<T>>;

int n, m;
vector<vector<int>> adj;
vector<int> ans;
unordered_set<int> pass;
min_pq<int> to_go;

void bfs(int s) {
    for (auto neighbour:adj[s]) {
        if (pass.count(neighbour)==0) {
            to_go.push(neighbour);
            pass.insert(neighbour);
        }
    }
    // cout << "Top" << endl;
    if (to_go.empty()) return;
    int next = to_go.top();
    // cout << "Pop" << endl;
    to_go.pop();
    ans.push_back(next);
    // if (to_go.empty()) return;
    bfs(next);
}

int main(int argc, char* argv[]) {
    cin >> n >> m;
    adj.assign(n+1, vector<int> {});
    int a, b;
    for (int i=0; i<m; i++) {
        cin >> a >> b;
        adj[a].push_back(b);
        if (a !=b) {
            adj[b].push_back(a);
        }
    }
    pass.insert(1);
    ans.push_back(1);
    // cout << adj[1].size() << endl;
    bfs(1);
    for (auto num : ans) {
        cout << num << " ";
    }
    cout << endl;
}