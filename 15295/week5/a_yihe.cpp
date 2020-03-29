    #include <cmath>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <iostream>
    #include <vector>
    #include <string>
    #include <queue>
    #include <algorithm>
    #include <functional>
    using namespace std;
    const int maxn = 510;
    const int dx[4] = { -1, 1, 0, 0 };
    const int dy[4] = { 0, 0, -1, 1 };
    char g[maxn][maxn];
    bool mark[maxn][maxn];
    int  p[maxn][maxn], deg[maxn][maxn];
    bool vis[maxn][maxn];
    queue<int> Q;
    int n, m, k;
     
    void dfs(int x, int y, int f) {
    	vis[x][y] = true;
    	p[x][y] = f;
    	for(int i = 0; i < 4; ++i) {
    		int tx = x + dx[i];
    		int ty = y + dy[i];
    		if(tx >= 0 && tx < n && ty >= 0 && ty < m && g[tx][ty] == '.' && !vis[tx][ty]) {
    			dfs(tx, ty, x*m+y);
    			deg[x][y]++;
    		}
    	}
    	return ;
    }
     
    int main() {
     
        //freopen("aa.in", "r", stdin);
     
    	scanf("%d %d %d", &n, &m, &k);
    	for(int i = 0; i < n; ++i) {
    		scanf("%s", g[i]);
    	}
    	memset(mark, false, sizeof(mark));
    	memset(vis, false, sizeof(vis));
    	memset(deg, 0, sizeof(deg));
    	for(int i = 0; i < n; ++i) {
    		for(int j = 0; j < m; ++j) {
    			if(!vis[i][j] && g[i][j] == '.') {
    				dfs(i, j, -1);
    			}
    		}
    	}
    	int cnt = 0;
    	for(int i = 0; i < n; ++i) {
    		for(int j = 0; j < m; ++j) {
    			if(g[i][j] == '.' && deg[i][j] == 0) {
    				Q.push(i*m+j);
    			}
    		}
    	}
     
    	while(cnt < k) {
    		int u = Q.front(); Q.pop();
    		int x = u / m, y = u % m;
    		mark[x][y] = true;
    		cnt++;
    		if(--deg[p[x][y]/m][p[x][y]%m] == 0) {
    			Q.push(p[x][y]);
    		}
    	}
    	for(int i = 0; i < n; ++i) {
    		for(int j = 0; j < m; ++j) {
    			if(mark[i][j]) {
    				printf("X");
    			} else {
    				printf("%c", g[i][j]);
    			}
    		}
    		printf("\n");
    	}
    	return 0;
    }
