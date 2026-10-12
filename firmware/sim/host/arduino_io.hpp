#pragma once

#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

class String : public std::string
{
public:
    using std::string::string;
    String(const std::string &value) : std::string(value) {}
    String substring(int begin, int end) const { return substr(static_cast<size_t>(begin), static_cast<size_t>(end - begin)); }
    float toFloat() const { return std::strtof(c_str(), nullptr); }
};

class Print
{
public:
    std::string output;
    template <typename T> void print(const T &value)
    {
        std::ostringstream stream;
        stream << value;
        output += stream.str();
    }
    void print(float value, int precision)
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(precision) << value;
        output += stream.str();
    }
    void println(const char *value)
    {
        print(value);
        output += '\n';
    }
};

extern Print Serial;

inline void delayMicroseconds(unsigned int /*duration*/) {}
