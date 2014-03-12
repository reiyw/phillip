#ifndef INCLUDE_HENRY_HENRY_INLINE_H
#define INCLUDE_HENRY_HENRY_INLINE_H


namespace phil
{


inline phillip_main_t *phillip_main_t::get_instance()
{
    static phillip_main_t singleton;
    return &singleton;
}


inline phillip_main_t::phillip_main_t()
    : m_lhs_enumerator(NULL), m_ilp_convertor(NULL), m_ilp_solver(NULL),
      m_kb(NULL), m_obs(NULL), m_lhs(NULL), m_ilp(NULL),
      m_timeout(-1), m_verboseness(0), m_is_debugging(false),
      m_clock_for_enumeration(0), m_clock_for_convention(0),
      m_clock_for_solution(0), m_clock_for_infer(0) {}


inline const lhs_enumerator_t* phillip_main_t::lhs_enumerator() const
{ return m_lhs_enumerator; }


inline const ilp_converter_t* phillip_main_t::ilp_convertor() const
{ return m_ilp_convertor; }


inline const ilp_solver_t* phillip_main_t::ilp_solver() const
{ return m_ilp_solver; }


inline const kb::knowledge_base_t* phillip_main_t::knowledge_base() const
{ return m_kb; }


inline void phillip_main_t::set_lhs_enumerator( const lhs_enumerator_t *ptr )
{ m_lhs_enumerator = ptr; }


inline void phillip_main_t::set_ilp_convertor( const ilp_converter_t *ptr )
{ m_ilp_convertor = ptr; }


inline void phillip_main_t::set_ilp_solver( const ilp_solver_t *ptr )
{ m_ilp_solver = ptr; }


inline void phillip_main_t::set_knowledge_base( kb::knowledge_base_t *kb )
{ m_kb = kb; }


inline void phillip_main_t::set_timeout( int t )
{ m_timeout = t; }


inline void phillip_main_t::set_verbose( int v )
{ m_verboseness = v; }


inline void phillip_main_t::set_debug_flag( bool flag )
{ m_is_debugging = flag; }


inline void phillip_main_t::set_param(
    const std::string &key, const std::string &param )
{ m_params[key] = param; }


inline void phillip_main_t::set_flag(const std::string &key)
{ m_flags.insert(key); }


inline const lf::logical_function_t* phillip_main_t::get_observation() const
{ return m_obs; }


inline const pg::proof_graph_t* phillip_main_t::get_latent_hypotheses_set() const
{ return m_lhs; }


inline const ilp::ilp_problem_t* phillip_main_t::get_ilp_problem() const
{ return m_ilp; }


inline const std::vector<ilp::ilp_solution_t>& phillip_main_t::get_solutions() const
{ return m_sol; }


inline const int& phillip_main_t::timeout() const
{ return m_timeout; }


inline const int& phillip_main_t::verbose() const
{ return m_verboseness; }


inline bool phillip_main_t::is_debugging() const
{ return m_is_debugging; }


inline const hash_map<std::string, std::string>& phillip_main_t::params() const
{ return m_params; }


inline const std::string& phillip_main_t::param( const std::string &key ) const
{
    static const std::string empty_str("");
    return has_key(m_params, key) ? m_params.at(key) : empty_str;
}


inline const hash_set<std::string>& phillip_main_t::flags() const
{ return m_flags; }


inline bool phillip_main_t::flag(const std::string &key) const
{ return m_flags.find(key) != m_flags.end(); }


inline const long& phillip_main_t::get_clock_for_flhs() const
{ return m_clock_for_enumeration; }


inline const long& phillip_main_t::get_clock_for_flpp() const
{ return m_clock_for_convention; }


inline const long& phillip_main_t::get_clock_for_fsol() const
{ return m_clock_for_solution; }


inline const long& phillip_main_t::get_clock_for_infer() const
{ return m_clock_for_infer; }


inline bool phillip_main_t::can_infer() const
{
    return
        (m_lhs_enumerator != NULL) and
        (m_ilp_convertor != NULL) and
        (m_ilp_solver != NULL);
}


inline void phillip_main_t::reset_for_inference()
{
    if( m_obs != NULL ) delete m_obs;
    if( m_lhs != NULL ) delete m_lhs;
    if( m_ilp != NULL ) delete m_ilp;
    m_obs = NULL;
    m_lhs = NULL;
    m_ilp = NULL;
    m_clock_for_enumeration = 0;
    m_clock_for_convention = 0;
    m_clock_for_solution = 0;
    m_clock_for_infer = 0;
    m_sol.clear();
}


}

#endif