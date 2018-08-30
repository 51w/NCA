#pragma once
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

#include "logging.hpp"

#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>;


namespace caffe {

using std::ostringstream;
using std::map;
using std::set;
using std::string;
using std::vector;
using std::shared_ptr;

}
