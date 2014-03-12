/* -*- coding: utf-8 -*- */

#pragma once

#include <sstream>
#include <cassert>


namespace phil
{


inline string_hash_t::string_hash_t( const std::string &s )
    : m_hash( get_hash(s) )
{}


inline string_hash_t string_hash_t::get_unknown_hash()
{
    char buffer[128];
    _sprintf(buffer, "_u%d", ms_issued_variable_count++ );
    return string_hash_t( std::string(buffer) );
}


inline unsigned string_hash_t::get_hash(std::string str)
{
    hash_map<std::string, unsigned>::iterator it = ms_hashier.find(str);
    if (it != ms_hashier.end())
        return it->second;
    else
    {
        ms_strs.push_back(str);
        unsigned idx(ms_strs.size() - 1);
        ms_hashier[str] = idx;
        return idx;
    }
}


inline string_hash_t& string_hash_t::operator = (const std::string &s)
{
    m_hash = get_hash(s);
    return *this;
}


inline string_hash_t& string_hash_t::operator = (const string_hash_t &h)
{
    m_hash = h.m_hash;
    return *this;
}


inline bool string_hash_t::operator > (const string_hash_t &x) const
{
    return m_hash > x.m_hash;
}


inline bool string_hash_t::operator < (const string_hash_t &x) const
{
    return m_hash < x.m_hash;
}


inline bool string_hash_t::operator == (const char *s) const
{
    return m_hash == ms_hashier.at(s);
}


inline bool string_hash_t::operator != (const char *s) const
{
    return not(*this == s);
}


inline bool string_hash_t::operator == (const string_hash_t &h) const
{
    return m_hash == h.m_hash;
}


inline bool string_hash_t::operator != (const string_hash_t &h) const
{
    return m_hash != h.m_hash;
}


inline bool string_hash_t::is_constant() const
{
    return std::isupper(this->string().at(0));
}


inline bool string_hash_t::is_unknown() const
{
    return startswith(this->string(), "_u");
}


inline literal_t::literal_t( const std::string &_pred, bool _truth )
    : predicate(_pred), truth(_truth) {}
    

inline literal_t::literal_t(
    predicate_t _pred, const std::vector<term_t> _terms, bool _truth )
    : predicate(_pred), terms(_terms), truth(_truth) {}


inline literal_t::literal_t(
    const std::string &_pred,
    const std::vector<term_t> _terms, bool _truth )
    : predicate(_pred), terms(_terms), truth(_truth) {}


inline literal_t::literal_t(
    const std::string &_pred,
    const term_t &term1, const term_t &term2, bool _truth )
    : predicate(_pred), truth(_truth)
{
    terms.push_back(term1);
    terms.push_back(term2);
}


inline literal_t::literal_t(
    const std::string &_pred,
    const std::string &term1, const std::string &term2,
    bool _truth )
    : predicate(_pred), truth(_truth)
{
    terms.push_back( string_hash_t(term1) );
    terms.push_back( string_hash_t(term2) );
}


inline bool literal_t::operator == (const literal_t &other) const
{
    if( truth != other.truth )               return false;
    if( predicate != other.predicate )       return false;
    if( terms.size() != other.terms.size() ) return false;
    for( size_t i=0; i<terms.size(); i++ )
    {
        if( terms[i] != other.terms[i] )
            return false;
    }
    return true;
}


inline std::string literal_t::to_string( bool f_colored ) const
{
    std::string exp;
    print( &exp, f_colored );
    return exp;
}


inline std::string literal_t::get_predicate_arity(
    bool do_distinguish_negation ) const
{
    std::string out = phil::format(
        "%s/%d", predicate.c_str(), (int)terms.size() );
    if( do_distinguish_negation and not truth ) out = "!" + out;
    return std::string( out );
}


inline void cdb_data_t::put(
    const void *key, size_t ksize, const void *value, size_t vsize)
{
    if (is_writable())
        m_builder->put(key, ksize, value, vsize);
}


inline const void* cdb_data_t::get(
    const void *key, size_t ksize, size_t *vsize) const
{
    return is_readable() ? m_finder->get(key, ksize, vsize) : NULL;
}


inline size_t cdb_data_t::size() const
{
    return is_readable() ? m_finder->size() : 0;
}


inline void print_console(const std::string &str)
{
    std::cerr << time_stamp() << str << std::endl;
}


inline void print_error(const std::string &str)
{
    std::cerr << " * ERROR * " << str << std::endl;
}


inline void print_warning(const std::string &str)
{
    std::cerr << " * WARNING * " << str << std::endl;
}


inline bool do_exist_file(const std::string &path)
{
    bool out(true);
    std::ifstream fin(path);
    if (not fin)
        out = false;
    else
        fin.close();
    return out;
}


inline std::string get_file_name( const std::string &path )
{
    int idx = path.rfind("/");
    return ( idx >= 0 ) ? path.substr(idx+1) : path;
}


inline size_t get_file_size(const std::string &filename)
{
    struct stat filestatus;
    stat( filename.c_str(), &filestatus );
    return filestatus.st_size;
}

  
inline size_t get_file_size( std::istream &ifs )
{
    size_t file_size =
        static_cast<size_t>( ifs.seekg(0, std::ios::end).tellg() );
    ifs.seekg(0, std::ios::beg);
    return file_size;
}


inline size_t string_to_binary( const std::string &str, char *out )
{
    size_t n(0);
    unsigned char size = static_cast<unsigned char>( str.size() );
    
    std::memcpy( out+n, &size, sizeof(unsigned char) );
    n += sizeof(unsigned char);

    std::memcpy( out+n, str.c_str(), sizeof(char) * str.size() );
    n += sizeof(char) * str.size();

    return n;
}


inline size_t num_to_binary( const int num, char *out )
{
    unsigned char n = static_cast<unsigned char>(num);
    std::memcpy( out, &n, sizeof(unsigned char) );
    return sizeof(unsigned char);
}


inline size_t bool_to_binary( const bool _bool, char *out )
{
    char c = _bool ? 1 : 0;
    std::memcpy( out, &c, sizeof(char) );
    return sizeof(char);
}


template <class T> inline size_t to_binary( const T &value, char *out )
{
    std::memcpy( out, &value, sizeof(T) );
    return sizeof(T);
}


inline size_t binary_to_string( const char *bin, std::string *out )
{
    size_t n(0);
    unsigned char size;
    char str[512];
    
    std::memcpy( &size, bin, sizeof(unsigned char) );
    n += sizeof(unsigned char);

    std::memcpy( str, bin+n, sizeof(char)*size );
    str[size] = '\0';
    *out = std::string(str);
    n += sizeof(char)*size;

    return n;
}


inline size_t binary_to_num( const char *bin, int *out )
{
    unsigned char num;
    std::memcpy( &num, bin, sizeof(unsigned char) );
    *out = static_cast<int>(num);
    return sizeof(unsigned char);
}


inline size_t binary_to_bool( const char *bin, bool *out )
{
    char c;
    std::memcpy( &c, bin, sizeof(char) );
    *out = ( c != 0 );
    return sizeof(char);
}


template <class T> inline size_t binary_to( const char *bin, T *out )
{
    size_t size = sizeof(T);
    std::memcpy( out, bin, size );
    return size;
}


template <class It, bool USE_STREAM> std::string join(
    const It &s_begin, const It &s_end, const std::string &delimiter)
{
    if (USE_STREAM)
    {
        std::ostringstream ss;
        for (It it = s_begin; it != s_end; ++it)
            ss << (it == s_begin ? "" : delimiter) << (*it);
        return ss.str();
    }
    else
    {
        std::string out;
        for (It it = s_begin; it != s_end; ++it)
            out += (it == s_begin ? "" : delimiter) + (*it);
        return out;
    }
}


template <class It> std::string join(
    const It &s_begin, const It &s_end,
    const std::string &fmt, const std::string &delimiter )
{
    std::string out;
    for (It it = s_begin; s_end != it; ++it)
    {
        std::string buf = format(fmt.c_str(), *it);
        out += (it != s_begin ? delimiter : "") + buf;
    }
    return out;
}



template <class T, class K> inline bool has_key( const T& map, const K& key )
{ return map.find(key) != map.end(); }


template <class T> bool has_intersection(
    const T &s1_begin, const T &s1_end,
    const T &s2_begin, const T &s2_end )
{
    for (T i1 = s1_begin; i1 != s1_end; ++i1)
        for (T i2=s2_begin; i2 != s2_end; ++i2 )
            if (*i1 == *i2) return true;
    return false;
}


template <class T> hash_set<T> intersection(
    const hash_set<T> &set1, const hash_set<T> &set2)
{
    bool set1_is_smaller = (set1.size() < set2.size());
    const hash_set<T> *smaller = (set1_is_smaller ? &set1 : &set2);
    const hash_set<T> *bigger = (set1_is_smaller ? &set2 : &set1);
    hash_set<T> out;

    for (auto it = smaller->begin(); it != smaller->end(); ++it)
    {
        if (bigger->find(*it) != bigger->end())
            out.insert(*it);
    }

    return out;
}

} // end phil

