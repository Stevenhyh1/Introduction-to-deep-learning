#include <iostream>
#include <vector>

typedef long long ll;

using namespace std;
vector<vector<int>> adj;

int main(int argc, char* argv[]) {
    int n;
    cin >> n;
    adj.assign(n, vector<int> (n, 0));
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            char cur;
            cin >> cur;
            if (cur == 'Y') {
                adj[i][j] = 1;
            }
        }
    }

    
}