#pragma once

#include <chrono>
#include <string>

class ScopeTimer {
public:
  ScopeTimer(const std::string &msg);
  ~ScopeTimer();

  template <typename Duration> double getElapsed() const {
    return std::chrono::duration_cast<Duration>(
               std::chrono::steady_clock::now() - m_start)
        .count();
  }

private:
  const std::string m_msg;
  const std::chrono::steady_clock::time_point m_start;
};

class Timer {
public:
  Timer(const std::string &msg);

  void start();
  void stop();
  void setMessage(const std::string &msg);

private:
  std::string m_msg;
  std::chrono::steady_clock::time_point m_start;
};
