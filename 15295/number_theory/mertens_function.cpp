#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <set>

using namespace std;

vector<int> fact;
vector<int> repeat;

int factor(int n) {
    // set<int> factors;
    int count = 0;
    while (n > 1) {
        int f = fact[n];
        // if (factors.find(f) != factors.end()) return 0;
        // factors.insert(f);
        count ++;
        n /= f;
    }
    if (count % 2 == 0) return 1;
    return -1;
}

int main(int argc, char * argv[]) {
    int n; 
    cin >> n;
    if (n==1) {
        cout << 1 << endl;
        return 0;
    }
    fact.assign(n+1, 0);
    repeat.assign(n+1, 1);
    for (int i=1; i<fact.size(); i++) {
        fact[i] = i;
    }
    for (int i=2; i*i <= n; i++) {
        if (fact[i] == i) {
            for (int j = i*i; j <= n; j += i) {
                fact[j] = i;
            }
            for (int j = 1; j*i*i <= n; j++) {
                repeat[j*i*i] = 0;
            }
        }
    }
    int res = 1;
    for (int i=2; i <= n; i++) {
        // cout << i << endl;
        res += factor(i) * repeat[i];
    }

    // for (int i=1; i<fact.size(); i++) {
    //     cout << i << ":" << fact[i] << endl;
    // }
    // cout << factor(25) << endl;

    // int res = 1;
    // for (int i = 2; i<=n; i++) {
    //     res += factor(i);
    // }
    cout << res << endl;
}