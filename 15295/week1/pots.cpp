#include <iostream>
#include <vector>
#include <stack>
#include <algorithm>
#include <queue>
#include <functional>

using namespace std;

int pots_count(vector<int> &pots) {
    int n = pots.size(), min_val = 0;
    sort(pots.begin(), pots.end());
    priority_queue<int,vector<int>, greater<int>> q;
    vector<int> result;
    for (int i=0; i<n; i++) {
        int cur = pots[i];
        if (q.empty() || q.top()==cur) {
            q.push(cur);
        }
        else
        {
            q.pop();
            q.push(cur);
        }
    }
    return(q.size());
}

int main() {
    int N;
    char ch;
    cin >> N;
    cin.get(ch);
    // cout << N;
    vector<int> pots(N, 0);
    for (int i=0; i<N; i++) {
        cin >> pots[i];
    }
    // cout << pots.back();
    int result = pots_count(pots);
    cout << result << endl;
}