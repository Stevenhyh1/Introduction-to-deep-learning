#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;

double a,b,c,d;

bool val (double cur) {
    double diag1max = max(max((a + cur) * (d + cur), (a + cur) * (d - cur)),max((a - cur) * (d + cur),(a - cur) * (d - cur)));
    double diag1min = min(min((a + cur) * (d + cur), (a + cur) * (d - cur)),min((a - cur) * (d + cur),(a - cur) * (d - cur)));
    double diag2max = max(max((b + cur) * (c + cur), (b + cur) * (c - cur)),max((b - cur) * (c + cur),(b - cur) * (c - cur)));
    double diag2min = min(min((b + cur) * (c + cur), (b + cur) * (c - cur)),min((b - cur) * (c + cur),(b - cur) * (c - cur)));
    if (diag1max < diag2min || diag2max < diag1min) {
        return false;
    }
    else
    {
        return true;
    }
    
}

int main(int argc, char *argv[]) {
    
    char ch;
    cin >> a >> b;
    cin.get(ch);
    cin >> c >> d;

    double low = 0, high = max(max(fabs(a),fabs(b)),max(fabs(c),fabs(d)));
    int t = 100;
    while (t--) {
        double mid = low + (high-low) / 2;
        if (val(mid)) {
            high = mid;
        }
        else
        {
            low = mid;
        }
    }
    double res = (low + high)/2;
    printf("%.10lf\n", res);
}