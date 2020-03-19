#include <iostream>
#include <vector>
#include <numeric>
#include <functional>

typedef long long ll;

using namespace std;

vector<ll> fact;

void precompute(ll max_n, ll m) {
    fact.assign(max_n+2, 0);
    fact[0] = fact[1] = 1;
    for (ll n=2; n <= max_n+1; n++) {
        fact[n] = (fact[n-1]*n) % m;
    }
}

ll expmod(ll a, ll b, ll m) {
    ll res = 1 % m;
    a = a % m;
    // cout << a << endl;
    while (b > 0) {
        if (b%2 == 1) res = res*a % m;
        a = a*a % m;
        b = b / 2;
    }
    return res;
}

ll inverse(ll x,ll m) {
    return expmod(x, m-2, m);
}

ll binomial(ll n, ll k, ll m) {
    return ((fact[n] * inverse(fact[n-k], m) % m) * inverse(fact[k],m)) % m;
}

int main (int argc, char* argv[]) {
    int n, k;
    cin >> n >> k;
    ll count;
    ll modulo = 1000000007;
    ll max_n = n+k-1;
    precompute(max_n, modulo);
    ll ans = binomial(n+k-1, k-1, modulo);
    cout << ans << endl;
}