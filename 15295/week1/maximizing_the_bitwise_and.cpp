#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <unordered_map>

using namespace std;

bool mycompare(int a, int b) {
    return a>b;
}

int main() {
    int n, cur;
    char ch;
    cin >> n;
    cin.get(ch);
    vector<int> nums;
    for (int i=0; i<n; i++) {
        cin >> cur;
        nums.push_back(cur);
    }
    sort(nums.begin(), nums.end(),mycompare);
    // cout << nums.size() << endl;
    unordered_map<int, vector<int>> hash;
    for (int i=0; i<n; i++) {
        int key = log2(nums[i]);
        hash[key].push_back(nums[i]);
    }
    
    // cout << hash[10].size() << endl;
    int result = 0; vector<int> single_nums;
    for (auto it=hash.begin(); it!=hash.end(); it++) {
        vector<int> cur = it->second;
        // cout << cur.size()<<endl;
        if (cur.size()>=2) {
            int res = cur[0] & cur[1];
            cout << res << endl;
            return 0;
        }
    }
}