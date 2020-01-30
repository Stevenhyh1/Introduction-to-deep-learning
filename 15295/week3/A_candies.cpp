#include<algorithm>
#include<iostream>
#include<string>
#include<map>
#include<set>
#include<vector>
//#include<cmath>
#include<stack>
#include<string.h>
#include<stdlib.h>
#include<cstdio>
#define maxn 103
#define INF 10000000
#define LL long long
using namespace std;
LL num; // global variable, the total number of candies
 
LL judge(LL k)
{
	LL n, sum;  //n: number of candies, sum: the candidies Vasya got
	sum = 0;
    n = num;
	while(n)
	{
		if(n<=k)   // if the number of remaining candies <=k, then Vasya can have them all
		{
			sum += n;
			break;
		}
		n -= k;
		sum += k;
		n -= n/10;  // Petya would have n/10 candies
	}
	return sum;  // the number of candies Vasya would got
}
int main(void)
{
	LL low, high, middle;
	cin >> num; //eg:63
	low= 1, high = num;
	while(low < high)
	{
		middle = (low + high)/2;
		if(judge(middle) >= (num + 1)/2)
			high = middle;
		else
			low = middle+1;
	}
	cout << low << endl;
	return 0;
}