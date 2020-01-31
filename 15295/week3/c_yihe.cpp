#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
    int n;
    char ch;
    cin >> n;
    cin.get(ch);
    vector<int> nums;
    for (int i=0; i<n; i++) {
        int cur;
        cin >> cur;
        nums.push_back(cur);
    }
    // cout << nums.back();
    
}