#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

vector<int> nums;
vector<int> deltas;

int gcd(int a, int b) {
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}

int main(int argc, char* argv[]) {
    int n, q;
    cin >> n >> q;

    for (int i=0; i<n; i++) {
        int num;
        cin >> num;
        nums.push_back(num);
    }

    for (int i=0; i<q; i++) {
        int num;
        cin >> num;
        deltas.push_back(num);
    }

    
    
    if (n == 1) {
        cout << accumulate(deltas.begin(), deltas.end(), nums[0]) << endl;
        return 0;
    }

    int res = nums[1]-nums[0];
    for (int i=2; i<nums.size(); i++) {
        int first = nums[i-1];
        int second = nums[i];
        res = -gcd(res, second-first);
    }
    // cout << res << endl;
    for (int i=0; i<deltas.size(); i++) {
        // cout << deltas[i] << endl;
        nums[0] += deltas[i];
        cout << gcd(nums[0], res) << endl;
        // cout << res << endl;
    }

    // int a = -24, b = 12, c = 6;
    // cout << gcd(a, gcd(b, c)) << endl;
    // cout << gcd(a, gcd(b-a, c-b)) << endl;
    
    
    return 0;
}