#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void row_swap(int a, int b, vector<vector<int>> &mat) {
    swap(mat[a-1],mat[b-1]);
}

void col_swap(int a, int b, vector<vector<int>> &mat) {
    for (int i=0; i<mat.size(); i++) {
        swap(mat[i][a-1], mat[i][b-1]);
    }
}

void mat_print(int a, int b, vector<vector<int>> &mat) {
    cout << mat[a-1][b-1] << endl;
}

int main() {
    int n, k;
    char ch;
    cin >> n >> k;
    cin.get(ch);
    vector<vector<int>> mat(n, vector<int> (n,0));
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cin >> mat[i][j];
        }
        cin.get(ch);
    }
    // cout << mat[1][2];
    for (int i=0; i<k; i++) {
        char operation;
        int first, second;
        cin >> operation >> first >> second;
        cin.get(ch);
        if (operation == 'R') row_swap(first, second, mat);
        else if (operation == 'C') col_swap(first, second, mat);
        else mat_print(first,second, mat);
    }
    return 0;
}