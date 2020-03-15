#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>

using namespace std;

int n;


int main(int argc, char *argv[]) {

    cin >> n;
    for (int i=0; i<n; i++) {
        int m;
        cin >> m;
        vector<int> nums(m,0);
        for (int j=0; j<m; j++) {
            int cur;
            cin >> cur;
            nums[j] = cur;
            // cout << cur << endl;
        }
        sort(nums.begin(), nums.end());
        unordered_set<int> unique;
        int count=0;
        for (int k=0; k<m; k++) {
            if (unique.find(nums[k]) == unique.end()) {
                // cout << nums[k] << endl;
                unique.insert(nums[k]);
                count++;
            }
        }
        // unique.clear();
        cout << count << endl;
    }

}