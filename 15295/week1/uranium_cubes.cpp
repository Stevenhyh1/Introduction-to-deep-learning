#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>

using namespace std;

int median_sum(const deque<int> &v) {
    int n = v.size(), median = n/2, result = 0;
    int num = v[median];
    for (int i=0; i<v.size(); i++) {
        result += abs(v[i]-num);
    }
    return result;
}

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
    int result = median_sum(dq);

    while (result>B) {
        cout << result << endl;
        int front_dis = result - dq.front(), back_dis = result-dq.back();
        if (front_dis > back_dis) {
            dq.pop_front();
        }
        else {
            dq.pop_back();
        }
        result = median_sum(dq);
    }
    cout << dq.size() << endl;
    return 0;
}