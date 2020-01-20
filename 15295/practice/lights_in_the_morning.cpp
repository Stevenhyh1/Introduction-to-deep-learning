#include <iostream>
#include <vector>

using namespace std;

bool ifgothrough(int N, int D, const vector<vector<int>> &tf_light) {
    // cout << D << endl;
    int loc, start, g, r;
    for (int i=0; i<N; i++) {
        vector<int> cur = tf_light[i];
        loc = cur[0];
        start = cur[1];
        g = cur[2];
        r = cur[3];
        if (start > D) continue;
        if (loc<start) return false;
        if ((loc-start)%(g+r)>g) {
            // cout << (loc-start)%(g+r) << endl;
            return false;
        }
    }
    return true;
}

int main() {
    int N, D;
    char ch;
    cin >> N >> D;
    cin.get(ch);
    // cout << N << D << endl;
    vector<vector<int>> tf_light(N, vector<int> (4, 0));
    for (int i=0; i<N; i++) {
        for (int j=0; j<4; j++) {
            cin >> tf_light[i][j];
        }
        // cout << tf_light[i][0] <<  tf_light[i][1] <<  tf_light[i][2] <<  tf_light[i][3] << endl;
        cin.get(ch);
    }

    if (ifgothrough(N,D,tf_light)) cout << "YES" << endl;
    else
    {
        cout << "NO" << endl;
    }
    return 0;
}