#pragma once

#include <cctype>
#include <cstdint>
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
    int indexOf(const char *needle) const
    {
        if (needle == nullptr) return -1;
        const size_t found = find(needle);
        return found == npos ? -1 : static_cast<int>(found);
    }
    float toFloat() const { return std::strtof(c_str(), nullptr); }
    int toInt() const { return static_cast<int>(std::strtol(c_str(), nullptr, 10)); }
};

class Print
{
public:
    std::string output;

    virtual ~Print() = default;

    virtual void write(uint8_t byte) { output += static_cast<char>(byte); }

    virtual void write(const uint8_t *buffer, std::size_t size)
    {
        output.append(reinterpret_cast<const char *>(buffer), size);
    }

    template <typename T> void print(const T &value)
    {
        std::ostringstream stream;
        stream << value;
        const std::string text = stream.str();
        write(reinterpret_cast<const uint8_t *>(text.data()), text.size());
    }
    void print(float value, int precision)
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(precision) << value;
        const std::string text = stream.str();
        write(reinterpret_cast<const uint8_t *>(text.data()), text.size());
    }
    void println(const char *value)
    {
        print(value);
        write('\n');
    }
    void println(const String &value) { println(value.c_str()); }
    void println(int value)
    {
        print(value);
        write('\n');
    }
    void println() { write('\n'); }
    void print(const String &value) { print(value.c_str()); }
};

inline void delayMicroseconds(unsigned int /*duration*/) {}
