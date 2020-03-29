#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace std;

pair<int,int> matching (vector<vector<int>> daughters, vector<int> princes) {
    int n = daughters.size();
    // for (int i=0; i<princes.size();i++) {
    //     cout << princes[i] << endl;
    // }
    
    vector<int> unmatchdaughter;
    for (int i=0; i<n; i++) {
        if (daughters[i].size()==0) {
            unmatchdaughter.push_back(i+1);
            continue;
        }
        for (int j=0; j<daughters[i].size(); j++) {
            int cur = daughters[i][j];
            if (princes[cur-1] == 1) {
                princes[cur-1]--;
                break;
            }
            if (j == daughters[i].size()-1) {
                unmatchdaughter.push_back(i+1);
            }
        }
    }
    
    if (accumulate(princes.begin(), princes.end(),0) == 0) {
        return make_pair(-1, -1);
    }
    // cout << accumulate(princes.begin(), princes.end(),0) << endl;
    for (int i=0; i<princes.size(); i++) {
        if (princes[i]==1) {
            // cout << unmatchdaughter.size() << endl;
            for (int j=0; j<unmatchdaughter.size(); j++) {
                if (daughters[unmatchdaughter[j]-1].size()==0) {
                    return make_pair(unmatchdaughter[j], i+1);
                }
                // cout << unmatchdaughter[j] << endl;
                int front = daughters[unmatchdaughter[j]-1].front();
                // cout << i << endl;
                // cout << front << endl;
                return make_pair(unmatchdaughter[j], i+1);

            }
        }
    }
    return make_pair(-1, -1);
}

int main(int argc, char* argv[]) {
    int t, n;
    vector<int> princes;
    vector<vector<int>> daughters;
    cin >> t;
    for (int i=0; i<t; i++) {
        cin >> n;
        // cout << n << endl;
        daughters.assign(n, vector<int> ());
        princes.assign(n, 1);
        for (int j=0; j<n; j++) {
            int k;
            cin >> k;
            if (k==0) continue;
            for (int l=0; l<k; l++) {
                int cur;
                cin >> cur;
                daughters[j].push_back(cur);
            }
        }
        pair<int,int> res = matching(daughters, princes);
        // cout << res.size() << endl;
        if (res.first==-1) {
            cout << "OPTIMAL" << endl;
        }
        else {
            cout << "IMPROVE" << endl;
            cout << res.first << " " << res.second << endl;        
        }
        daughters.clear();
        princes.clear();
    }

}