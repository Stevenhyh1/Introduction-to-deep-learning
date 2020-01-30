#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
    int n, x, y, c, num=1;
    cin >> n >> x >> y >> c;
    int left = y-1, right = n-y, up = x-1, down = n-x;
    int ru = right + up+1, rd = right + down+1, lu = left + up+1, ld = left + down+1;
    // cout << left << right << up << down << endl;
    // cout << ru << rd << lu << ld << endl;
    int dru = 0, drd = 0, dlu = 0, dld = 0;
    int res = 0;
    while (num < c) {
        res++;
        num += 4*(res);

        int dleft = left > 0 ? 0 : 2*(-left)+1;
        int dup = up > 0 ? 0 : 2*(-up)+1;
        int dright = right > 0 ? 0 : 2*(-right)+1;
        int ddown = down > 0 ? 0 : 2*(-down)+1;

        int dru = ru > 0 ? 0 : -ru + 1;
        int drd = rd > 0 ? 0 : -rd + 1;
        int dlu = lu > 0 ? 0 : -lu + 1;
        int dld = ld > 0 ? 0 : -ld + 1;
        
        // cout << dleft << dright << dup << ddown << endl;
        // cout << left << right << up << down << endl;
        // cout << dru << drd << dlu << dld << endl;
        // cout << ru << rd << lu << ld << endl;

        num = num - dleft - dup - dright - ddown;
        num = num + dru + drd + dlu + dld;

        left--;right--;up--;down--;
        ru--;rd--;lu--;ld--;

        // cout << "number" << num << endl;
    }
    cout << res << endl;
}