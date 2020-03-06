#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

int main(){
    int T;
    cin >> T;
    vector<long long> answer;
    for (int i = 0; i < T; i++){
        int n;
        cin >> n;
        int num_0_5 = 0;
        for (int i = 0; i < n; i++){
            double temp_p;
            cin >> temp_p;
            if (temp_p == 0.500000){
                num_0_5 ++;
            }
        }
        if (num_0_5 == 0) cout << 0 << endl;
        else{
            cout <<(long long) pow(2, n) - (long long) pow(2, n-num_0_5) << endl;
        }
    }
    return 0;
}