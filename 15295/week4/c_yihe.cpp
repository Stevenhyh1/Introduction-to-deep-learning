#include <iostream>
#include <vector>
#include <utility>

using namespace std;

int perform(const int k, const vector<int> nums, vector<pair<char,char>> &signs) {
    pair<int, int> dp = make_pair(0,0); //first is small, second is large
    int i=0;
    char first, second;
    while (i<nums.size()) {
        if (dp.first - nums[i] >= 0) {
            dp.first -= nums[i];
            first = '-';
        }
        else {
            dp.first = dp.second - nums[i];
            second = '-';
        }
        if (dp.second + nums[i] <= k) {
            dp.second += nums[i];
            second = '+';
        }
        else
        {
            dp.second = 
        }
        
        i++;
    }
    return i;
}

int main(int argc, char* argv[]) {
    int n, k;
    cin >> n >> k;
    vector<int> nums;
    vector<pair<char,char>> signs; //first for the smaller, second for the larger
    for (int i=0; i<n; i++) {
        int cur;
        cin >> cur;
        nums.push_back(cur);
    }
    // cout << nums.back() << endl;
    int res = perform(k, nums, signs);
    cout << res << endl;
    for (int i=0; i<res; i++) {
        cout << signs[i];
    }
    cout << endl;
}