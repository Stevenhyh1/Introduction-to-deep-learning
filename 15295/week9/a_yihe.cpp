#include <iostream>

using namespace std;

typedef long long ll;

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

int main(int argc, char *argv[]) {
    ll m, n;
    cin >> m >> n;
    ll modulo = 100003;
    ll ans = expmod(m, n, modulo) - m * expmod(m-1, n-1, modulo);
    ans = ans % modulo;
    if (ans < 0) {
        ans += modulo;
    }
    cout << ans << endl;
}

// Your question has been answered:

// See my previous announcement about mods and negatives.


// 50 9787

// Participant's output
// -48644

// Jury's answer
// 51359

// Checker comment
// wrong answer expected '51359', found '-48644'