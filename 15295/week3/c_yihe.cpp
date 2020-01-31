#include <iostream>
#include <vector>

using namespace std;

class matrix {
    
    public:
    double a, b, c, d;
    double matmax() {
        return max(max(a,b),max(c,d));
    }

};

int main(int argc, char *argv[]) {
    matrix A,B;
    char ch;
    cin >> A.a >> A.b;
    cin.get(ch);
    cin >> A.c >> A.d;

    double low = 0, high = A.matmax();
    // cout << high << endl;

}