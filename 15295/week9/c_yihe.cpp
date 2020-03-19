#include <iostream>
#include <vector>

typedef long long ll;

using namespace std;
vector<vector<char>> adj;

int main(int argc, char* argv[]) {
    int n;
    cin >> n;
    vector<vector<char>> mat(n, vector<char> (n, ' '));
    adj.assign(n, vector<char> ());
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            char cur;
            cin >> cur;
            mat[i][j] = cur;
        }
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<i; j++) {
            char cur = mat[i][j];
            // cout << cur << endl;
            if (cur == 'N') {
                adj[j].push_back(i);
            }
            else if (cur == 'Y')
            {
                adj[i].push_back(j);
            }
        }
    }
    ll count = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<adj[i].size(); j++) {
            int second = adj[i][j];
            // cout << "Second " << second;
            for (int k=0; k<adj[second].size(); k++) {
                int third = adj[second][k];
                for (int l=0; l<adj[third].size(); l++) {
                    int fourth = adj[third][l];
                    if (fourth==i) count++;
                }
                // cout << "Third " << third << endl;
            }
        }
    }
    count /= 3;
    cout << count << endl;
}