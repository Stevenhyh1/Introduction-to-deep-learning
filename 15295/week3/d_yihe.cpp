#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;

int n;
int pos;
vector<double> buoys;
double eps = 1e-15;

//vector index starts from 0

double myabs(double now)
{
	return now>=0 ? now : -now;
}

double cal(double cur) {
    double sum = 0, res = numeric_limits<int>::max();
    for (int i=0; i<n; i++) {
        sum = 0;
        for (int j=0; j<i; j++) {
            sum += myabs(buoys[i] - buoys[j] - (i-j)*cur);
        }
        for (int j=i+1; j<n; j++) {
            sum += myabs(buoys[j] - buoys[i] - (j-i)*cur);
        }
        if (sum<res) {
            res = sum;
            pos = i;
        }
    }
    return res;
}

int main(int argc, char* argv[])  {
    ifstream input;
    input.open("../input.txt");
    if (input.is_open()) {
        input >> n;
        for (int i=0; i<n; i++) {
            double cur;
            input >> cur;
            buoys.push_back(cur);
        }
        input.close();
    }
    else
    {
        cout << "Unable to open file" << endl;
        exit(1);
    }
    double low = 0, high = 1e6;
    int t=100;
    while (t--) {
    // while (myabs(high-low)>eps) {
        double m1 = low + (high-low) / 2;
        double m2 = m1 + (high-m1) /2 ;
        if (cal(m1) > cal(m2)) {
            low = m1;
        }
        else
        {
            high = m2;
        }
    }
    double dis = (low+high)/2;
    double res = cal(dis);

    vector<double> result(n,0);
    for (int i=0; i<pos; i++) {
        result[i] = buoys[pos] - dis*(pos-i);
    }
    for (int i=pos+1; i<n; i++) {
        result[i] = buoys[pos] + dis*(i-pos);
    }
    result[pos] = buoys[pos];

    freopen("../output.txt", "w", stdout);

    printf("%.4f\n", res);
    printf("%.4f\n",dis);
    printf("%.d\n",pos);
    for (int i=0; i<n; i++) {
        printf("%.10lf ", result[i]);
    }

}
