#include <iostream>
#include <vector>
#include <queue>

using namespace std;

vector<vector<int>> adj;
int n;

int main(int argc, char *argv[]) {
    cin >> n;
    adj.assign(n, vector<int> {});
    for (int i=0; i<n; i++) {
        int m;
        cin >> m;
        for (int j=0; j<m; j++) {
            int val;
            cin >> val;
            adj[i].push_back(val);
        }
    }
    
    // cout << adj[5][2] << endl;
}
