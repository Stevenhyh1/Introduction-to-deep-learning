#include <iostream>

using namespace std;

typedef unsigned long long ll;

ll val(ll num, ll k) {
    ll n, sum; 
	sum = 0;
    n = num;
	while(n)
	{
		if(n<=k)
		{
			sum += n;
			break;
		}
		n -= k;
		sum += k;
		n -= n/10; 
	}
	return sum;
}

int main (int argc, char *argv[]) {
    ll n;
    cin >> n;
    ll low = 1, high = n;
    while (low < high) {
        ll mid = low + (high - low) / 2; 
        if (val(n,mid) >= (n+1)/2) {
            high = mid;
        }
        else {
            low = mid+1;
        }
    }
    cout << low << endl;
}