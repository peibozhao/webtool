#pragma once

#include "httplib.h"

class Service {
public:
  Service(httplib::Server &http_server);

  virtual ~Service() {}
};
