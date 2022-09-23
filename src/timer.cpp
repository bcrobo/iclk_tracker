#include <iclk/timer.hpp>
#include <iostream>

ScopeTimer::ScopeTimer(const std::string &msg)
    : m_msg(msg), m_start(std::chrono::steady_clock::now()) {}

ScopeTimer::~ScopeTimer() {
  const auto elapsed = std::chrono::steady_clock::now() - m_start;
  std::cout
      << m_msg << ": "
      << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()
      << " us\n";
}

Timer::Timer(const std::string &msg) : m_msg(msg) {}

void Timer::setMessage(const std::string &msg) { m_msg = msg; }

void Timer::start() { m_start = std::chrono::steady_clock::now(); }

void Timer::stop() {
  const auto elapsed = std::chrono::steady_clock::now() - m_start;
  std::cout
      << m_msg << ": "
      << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()
      << " us\n";
}
