#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
    int T;
    cin >> T;
    for (int i=0; i<T; i++) {
        int n;
        cin >> n;
        // cout << n << endl;
        for (int j=0; j<n; j++) {
            float num;
            cin >> num;
            // cout << num << endl;
            if (num==0.500000) {
                cout << n << endl;
                string dummy;
                getline(cin, dummy);
                break;
            }   
        }
    }
}