// Solution by team:
// - Marcin Wojnarowski (376886)
// - Jonathan Arnoult (369910)
// - Emilien Ganier (369941)

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

using namespace std;

int hammingDist(const vector<int>& x, const vector<int>& y, int dim) {
  int count = 0;
  for (int i = 0; i < dim; i++) {
    if (x[i] != y[i]) {
      count++;
    }
  }

  return count;
}

void printVector(const vector<int>& x) {
  for (int i = 0; i < x.size(); i++) {
    cout << x[i] << (i != x.size() - 1 ? " " : "");
  }
  cout << endl;
  cout.flush();
}

/* reports a solution using the format demanded by the Codeforces problem */
void reportSolution(const vector<int>& x) {
  cout << "* ";
  cout.flush();
  printVector(x);
}

/* to be used when running locally */
class offlineANNS {
 private:
  int d;
  int n;
  int r;
  double c;
  vector<int> z;
  vector<vector<int>> P;

  int k;
  int l;
  vector<vector<int>> h;
  vector<map<vector<int>, vector<int>>> T;

 public:
  offlineANNS(int dim,
              int radius,
              double approx,
              int numPoints,
              vector<int> center) {
    d = dim;
    r = radius;
    c = approx;
    n = numPoints;
    z = center;

    P = vector<vector<int>>(n, vector<int>(d, 0));

    for (int i = 0; i < d; i++) {
      P[0][i] = z[i];
    }

    /* read the rest of the dataset */
    for (int j = 1; j < n; j++) {
      for (int i = 0; i < d; i++) {
        cin >> P[j][i];
      }
    }

    /* initialize the data structure by sampling the hash functions and filling
     * the corresponding hash tables */
    l = ceil(pow(n, log(1.0 - double(r) / double(d)) /
                        log(1.0 - c * double(r) / double(d))) *
             log(n));
    k = ceil(log(n) / log(1 / (1.0 - c * double(r) / double(d))));

    h = vector<vector<int>>(l, vector<int>(k, -1));

    for (int u = 0; u < l; u++) {
      for (int v = 0; v < k; v++) {
        h[u][v] = rand() % d;
        cout << h[u][v] << endl;
      }
    }

    T = vector<map<vector<int>, vector<int>>>(l,
                                              map<vector<int>, vector<int>>());

    for (int u = 0; u < l; u++) {
      for (int j = 0; j < n; j++) {
        vector<int> pHash = vector<int>(k, -1);
        for (int v = 0; v < k; v++) {
          pHash[v] = P[j][h[u][v]];
        }

        map<vector<int>, vector<int>>::iterator it = T[u].find(pHash);

        if (it != T[u].end()) {
          (it->second).push_back(j);
        } else {
          T[u][pHash] = vector<int>(1, j);
        }
      }
    }
  }

  /* issues a query to the ANNS data structure and returns the answer */
  optional<vector<int>> query(const vector<int>& q) {
    int found = -1;
    for (int u = 0; u < l && found == -1; u++) {
      vector<int> qHash = vector<int>(k, -1);
      for (int v = 0; v < k; v++) {
        qHash[v] = q[h[u][v]];
      }

      map<vector<int>, vector<int>>::iterator it = T[u].find(qHash);
      if (it != T[u].end()) {
        int size = (it->second).size();
        for (int e = 0; e < size && found == -1; e++) {
          int j = (it->second)[e];
          if (hammingDist(q, P[j], d) <= c * r) {
            found = j;
          }
        }
      }
    }

    if (found == -1) {
      return nullopt;
    } else {
      auto a = vector<int>(d, -1);
      for (int i = 0; i < d; i++) {
        a[i] = P[found][i];
      }
      return a;
    }
  }
};

/* to be used for submitting to Codeforces */
class onlineANNS {
 private:
  int d;  // dimension of the hypercube

 public:
  onlineANNS(int dim) { d = dim; }

  /* issues a query using the format demanded by the Codeforces problem, reads
   * the answer and returns it */
  optional<vector<int>> query(const vector<int>& q) {
    cout << "q ";
    cout.flush();
    printVector(q);

    vector<int> answer = vector<int>();
    int size;

    cin >> size;
    if (size == 1) {
      return nullopt;
    }

    for (int i = 0; i < size; i++) {
      int elem;
      cin >> elem;
      answer.push_back(elem);
    }

    return answer;
  }
};

int main() {
  srand(time(NULL));
  std::random_device rd;
  std::mt19937 g(rd());

  int d;
  int r;
  double c;
  int n;
  int N;

  /* read parameters */
  cin >> d;
  cin >> r;
  cin >> c;
  cin >> n;
  cin >> N;

  /* read center point */
  auto z = vector<int>(d, 0);
  for (int i = 0; i < d; i++) {
    cin >> z[i];
  }

  offlineANNS ds(d, r, c, n, z);
  // onlineANNS ds(d);

  int mu =
      min(r, static_cast<int>(ceil(2.0 * exp(1) * exp(1) * (log(n) + 1.0))));

  for (size_t _ = 0; _ < N; _++) {
    // sample q
    vector<int> indices(d);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), g);

    vector<int> q = z;
    for (int i = 0; i < r - mu; i++) {
      q[indices[i]] = 1 - q[indices[i]];
    }

    while (ds.query(q).has_value() && hammingDist(q, z, d) < r) {
      int w = static_cast<int>(ceil(c * r)) + 1 - hammingDist(q, z, d);

      // sample I
      vector<int> I;
      for (int i = 0; i < d; i++) {
        if (q[i] == z[i]) {
          I.push_back(i);
        }
      }
      shuffle(I.begin(), I.end(), g);
      I.resize(w);

      // compute u's
      vector<vector<int>> u;
      u.push_back(q);
      for (size_t i = 0; i < w; i++) {
        auto previous = u.back();
        previous[I[i]] = 1 - previous[I[i]];
        u.push_back(previous);
      }

      // find j*
      for (size_t i = 1; i < u.size(); i++) {
        auto has = ds.query(u[i]).has_value();

        if (!has) {
          // TODO: what happens when j* = 0?
          if (i != 1) {
            q[I[i - 2]] = 1 - q[I[i - 2]];
          }
          break;
        }
      }
    }

    if (!ds.query(q).has_value() && hammingDist(q, z, d) <= r) {
      reportSolution(q);
      return 0;
    }
  }

  return 1;
}
