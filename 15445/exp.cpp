// test.cpp
#include <iostream>
using namespace std;
int main()
{
  cout << __cplusplus << endl;

  int global_depth_ = 0;
  int mask = (1 << global_depth_) - 1;
  int index = std::hash<int>()(0) & mask;
  cout << index << " " << mask << " " << (1 << global_depth_) << endl;

  global_depth_ = 1;
  mask = (1 << global_depth_) - 1;
  cout << mask << " " << (1 << global_depth_) << endl;

  global_depth_ = 2;
  mask = (1 << global_depth_) - 1;
  cout << mask << " " << (1 << global_depth_) << endl;

  return 0;
}