#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <unordered_map>

using namespace std;

int gcd(int a, int b) {
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}

int lcm(int a, int b) {
    if (a == 0 && b == 0) {
        return 0;
    }
    int lcm_val = std::abs(a * b) / gcd(a, b);
    return lcm_val;
}

bool isprime(int a) {
    if (a == 1) return false;
    for (int d = 2; d * d <= a; d++) {
        if (a % d == 0) {
            return false;
        }
    }
    return true;
}

std::vector<bool> sieve(int n) {
    std::vector<bool> is_prime (n+1, true);
    is_prime[0] = is_prime[1] = false;
    for (int i=2; i * i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i*i; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }
    return is_prime;
}

int factor(int n) {
    std::unordered_map<int, int> map;
    for (int i=2; i*i <= n; i++) {
        while (n % i == 0) {
            map[i]++;
            n /= i;
        }
    }
    if (n > 1) map[n]++;
    int res = 1;
    for (auto it = map.begin(); it != map.end(); it++) {
        // cout << it->first << ":" << it->second << endl;
        res *= (it->second + 1);
    }
    return res;
}

int main(int argc, char *argv[]) {
    int x;
    std::cin >> x;
    if (x == 1) {
        std::cout << 1 << std::endl;
        return 0;
    }
    if (x == 2) {
        std::cout << 3 << std::endl;
        return 0;
    }

    factor(x);
    // std::vector<int> factors = factor(x);
    int res = 3;
    for (int i=3; i<=x; i++) {
        if (isprime(i)) {
            res += 2;
        }
        else {
            res += factor(i);
        }
    }
    // for (int i=0; i < factors.size(); i++) {
    //     std::cout << factors[i] << std::endl;
    // }
    // 2:3, 3:5, 4:8, 5:10, 6:14, 7:16, 8:20, 9:23, 10:27, 11:29, 12:35, 13:37, 14:41, 15:45, 16:50, 17:52, 18:58, 19:60, 20:66, 21:70, 22:74, 23:76, 24:84, 25:87, 26:91, 27:95, 28:101, 29:103, 30:111, 31:113, 32:119
    // std::vector<int> a {1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6, 2, 4, 4, 5, 2, 6, 2, 6, 4, 4, 2, 8, 3, 4, 4, 6, 2, 8, 2, 6};
    // for (int i=1; i<a.size(); i++) {
    //     a[i] += a[i-1];
    //     std::cout << i+1 << ":" << a[i] << ", ";
    // }
    
    std::cout << res << std::endl;
    return 0;
}