#include "hand.h"

int default_value(int a, int b) {
    if (a <= 0) {
        return b;
    } else {
        return a;
    }
}

int startswith(std::string s, std::string sub) {
    return s.find(sub) == 0 ? 1 : 0;
}

int endswith(std::string s, std::string sub) {
    return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}