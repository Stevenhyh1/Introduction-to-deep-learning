#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>

using namespace std;

int main() {
    int n, L, B;
    char ch;
    cin >> n >> L >> B;
    cin.get(ch);
    vector<int> line(n,0);
    for (int i=0;i<n; i++) {
        cin >> line[i];
        cin.get(ch);
    }
    sort(line.begin(), line.end());
    deque<int> dq(line.begin(), line.end());
    int result=0, median=dq.size()/2;
    int num = dq[median];
    for (int i=0;i<dq.size(); i++) {
        result += abs(dq[i] - num);
    }
    // cout << result;
    bool even;
    while (result>B) {
        cout << result << endl;
        even = (dq.size() % 2 == 0);
        int cur_med = dq[median];
        int front_dis = cur_med - dq.front(), back_dis = cur_med-dq[median];
        if (front_dis > back_dis) {
            result -= front_dis;
            dq.pop_front();
            median--;
        }
        else {
            result -= back_dis;
            dq.pop_back();
        }
        if (even) {
            median++;
            result -= dq[median] - cur_med;
        }
    }
    cout << dq.size() << endl;
    return 0;
}