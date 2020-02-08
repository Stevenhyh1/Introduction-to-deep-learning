//#include<bits/stdc++.h>
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
#include <utility>      // std::pair, std::make_pair
#include <string>       // std::string
#include <iostream>     // std::cout
#define maxn 103
#define INF 10000000
#define LL long long

using namespace std;

#pragma GCC optimize("Ofast,no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
#pragma GCC optimize("unroll-loops")

/// Typedef
typedef long long ll;

#define sc1(a) scanf("%lld",&a)
#define sc2(a,b) scanf("%lld %lld",&a,&b)

#define pf1(a) printf("%lld\n",a)
#define pf2(a,b) printf("%lld %lld\n",a,b)

#define mx 100007
#define mod 100000007
#define PI acos(-1.0)

#define size1
#define pb push_back

ll dp1[mx], dp2[mx];


std::pair <ll,ll> first;
std::pair <ll,ll> second;
ll result;
ll result_prev;
vector<ll> height1;
vector<ll> height2;

int main()
{
    ll N;
    cin >> N;
    height1.resize(N);
    height2.resize(N);
    for(ll i=0;i<N;i++){
      cin >> height1[i];
    }
    for(ll i=0;i<N;i++){
      cin >> height2[i];
    }
    if (N==1){
       result = max(height1[0],height2[0]);
       cout << result;
       return 0;
    }
    if (N==2){
       result = max(height1[0]+height2[1], height1[1]+height2[0]);
       cout << result;
       return 0;
    }
    first = std::make_pair(height1[0],height2[0]);
    second = std::make_pair(height1[1]+height2[0], height2[1]+height1[0]);
    for (ll i = 2; i < N; i++){
        ll second_max = max(first.second, second.second);
        ll first_max = max(first.first, second.first);
        // int result_prev = result;
        result = max(height1[i]+ second_max, height2[i]+first_max);
        first = second;
        second = std::make_pair(height1[i]+ second_max, height2[i]+first_max);
    }
    cout <<result << endl;
    
    // ll k, num, prime, m, tc, t = 1;
    // sc1(num);
    // ll arr[num + 5], brr[num + 5];

    // for(ll i = 1; i <= num; i++) sc1(arr[i]);
    // for(ll i = 1; i <= num; i++) sc1(brr[i]);

    // for(ll i = 1; i <= num; i++){
    //     dp1[i] = dp2[i - 1] + arr[i];
    //     dp2[i] = dp1[i - 1] + brr[i];
    //     dp1[i] = max(dp1[i], dp1[i - 1]);
    //     dp2[i] = max(dp2[i], dp2[i - 1]);
    // }

    // ll ans = 0;
    // for(ll i = 1; i <= num; i++){
    //     ans = max(ans, max(dp1[i], dp2[i]));
    // }
    // pf1(ans);
}