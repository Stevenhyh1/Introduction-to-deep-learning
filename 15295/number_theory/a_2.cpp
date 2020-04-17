#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <unordered_map>

using namespace std;

vector<int> fact;

int factor(int n) {
    unordered_map<int,int> factors;
    int res = 1;
    while (n > 1) {
        int f = fact[n];
        factors[f] ++;
        n /= f;
    }
    for (auto it=factors.begin(); it != factors.end(); it++) {
        res *= (it->second + 1);
    }
    return res;
}

int main(int argc, char * argv[]) {
    int n; 
    cin >> n;
    if (n==1) {
        cout << 1 << endl;
        return 0;
    }
    fact.assign(n+1, 0);
    for (int i=1; i<fact.size(); i++) {
        fact[i] = i;
    }
    for (int i=2; i*i <= n; i++) {
        if (fact[i] == i) {
            for (int j = i*i; j <= n; j += i) {
                fact[j] = i;
            }
        }
    }
    int res = 1;
    for (int i = 2; i<=n; i++) {
        res += factor(i);
    }
    cout << res << endl;
    
    // for (int i=0; i<fact.size(); i++) {
    //     cout << fact[i] << endl;
    // }

}