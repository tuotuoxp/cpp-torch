#pragma once

#include <iostream>
#include <iterator>
#include <assert.h>


template <class TIterator>
class Joiner
{
public:
    Joiner(const TIterator begin, const TIterator end, const std::string &deli) : begin_(begin), end_(end), deli_(deli) {}

    friend std::ostream& operator << (std::ostream& stream, const Joiner &j)
    {
        bool first = true;
        for (auto it = j.begin_; it != j.end_; it++)
        {
            if (!first)
            {
                stream << j.deli_;
            }
            stream << *it;
            first = false;
        }
        return stream;
    }

protected:
    const TIterator begin_, end_;
    const std::string &deli_;
};


template <class TIterator>
Joiner<TIterator> join(const TIterator begin, const TIterator end, const std::string &deli)
{
    return Joiner<TIterator>(begin, end, deli);
}

template <class TContainer>
Joiner<typename TContainer::const_iterator> join(const TContainer &c, const std::string &deli)
{
    return Joiner<typename TContainer::const_iterator>(c.begin(), c.end(), deli);
}


//////////////////////////////////////////////////////////////////////////


class asserter
{
public:
    asserter(bool expr) : expr_(expr) {}
    ~asserter() { assert(expr_); }

    template <class T>
    asserter& operator << (const T &j)
    {
        if (!expr_)
        {
            std::cerr << j;
        }
        return *this;
    }

protected:
    bool expr_;
};
