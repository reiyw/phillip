/* -*- coding: utf-8 -*- */

#ifndef INCLUDE_HENRY_LP_PROBLEM_H
#define INCLUDE_HENRY_LP_PROBLEM_H


#include <string>
#include <climits>

#include "./define.h"
#include "./proof_graph.h"

#define DO_REDUCE_ILP_ENTITIES


namespace phil
{

/** A namespace about linear-programming-problems. */
namespace ilp
{


typedef index_t variable_idx_t;
typedef index_t constraint_idx_t;

class variable_t;
class constraint_t;
class ilp_problem_t;
class ilp_solution_t;


enum constraint_operator_e
{
    OPR_UNDERSPECIFIED,
    OPR_EQUAL,
    OPR_LESS_EQ,
    OPR_GREATER_EQ,
    OPR_RANGE
};


/** A class of a variable in objective-function of ILP-problems. */
class variable_t
{
public:
    inline variable_t(const std::string &name, double coef);

    inline void set_coefficient( double coef );
    
    inline const std::string& name() const;
    inline double objective_coefficient() const;
    
    inline std::string to_string() const;

private:
    std::string m_name;
    /** Its coefficient in the objective function. */
    double m_objective_coefficient;
};


/** A class to express a constraint in ILP-problems. */
class constraint_t
{
public:
    struct term_t
    {
        variable_idx_t var_idx; /// Index of ilp_problems_t::m_variables.
        double coefficient;
    };

    inline constraint_t();
    inline constraint_t( const std::string &name, constraint_operator_e opr );
    inline constraint_t(
        const std::string &name, constraint_operator_e opr, double val );
    inline constraint_t(
        const std::string &name, constraint_operator_e opr,
        double val1, double val2 );

    inline bool is_empty() const;
    inline void add_term( variable_idx_t var_idx, double coe );
  
    inline bool is_satisfied(
        const std::vector<double> &lpsol_optimized_values ) const;
    
    inline const std::string& name() const;
    inline constraint_operator_e operator_type() const;
    inline const std::vector<term_t>& terms() const;
    inline double bound() const;
    inline double lower_bound() const;
    inline double upper_bound() const;

    inline void set_bound( double lower, double upper );
    inline void set_bound( double target );
    
    void print(
        std::string *p_out,
        const std::vector<variable_t> &var_instances) const;

    inline std::string to_string( const std::vector<variable_t> &vars ) const;
    
private:    
    inline bool _is_satisfied( double sol ) const;
    
    std::string m_name;
    constraint_operator_e m_operator;
    std::vector<term_t> m_terms;

    /** Value of Left-hand-side and right-hand-side of this constraint. */
    double m_target[2];
};


/** A class of ILP-problem. */
class ilp_problem_t
{
public:
    static const int INVALID_CUT_OFF = INT_MIN;

    inline ilp_problem_t( const pg::proof_graph_t* lhs );

    /** Add new variable to the objective-function.
     *  @return The index of added variable in m_variables. */
    inline variable_idx_t add_variable( const variable_t &var );

    /** Add new constraint.
     *  @return The index of added constraint in m_constraints. */
    inline constraint_idx_t add_constraint( const constraint_t &con );

    /** Add new variable of the given node.
     *  @return The index of added variable in m_variables. */
    variable_idx_t add_variable_of_node(
        pg::node_idx_t idx, double coef = 0.0 );

    /** Add new variable of the hypernode and related constraints.
     *  On calling this method, it is required that
     *  variables of nodes in the hypernode have been created.
     *  @return The index of added variable in m_variables. */
    variable_idx_t add_variable_of_hypernode(
        pg::hypernode_idx_t idx, double coef = 0.0,
        bool do_add_requisite_variable = false );

    /** Add constraint for dependency between the target node
     *  and hypernodes which have the node as its element.
     *  On calling this method, it is required that
     *  variables of target node and related hypernodes have been created.
     *  @return True when new constraints added. */
    constraint_idx_t add_constraint_of_dependence_of_node_on_hypernode(
        pg::node_idx_t idx, bool do_add_requisite_variable = false);

    /** Add constraint for dependency
     *  between the target hypernode and its parents.
     *  On calling this method, it is required that
     *  variables of related hypernodes have been created.
     *  @return The index of added constraint. */
    constraint_idx_t add_constraint_of_dependence_of_hypernode_on_parents(
        pg::hypernode_idx_t, bool do_add_requisite_variable = false);

    /** Add constraint for mutual-exclusion between terms.
     *  On calling this method, it is required that
     *  variables of related unification-nodes have been created.
     *  @return The index of added constraint. */
    constraint_idx_t add_constraint_of_mutual_exclusion(
        pg::node_idx_t n1, pg::node_idx_t n2, const pg::unifier_t &uni,
        bool do_add_requisite_variable = false );
    void add_constraints_of_mutual_exclusions(
        bool do_add_requisite_variable = false );

    /** Add constraints about transitivity of unifications.
     *  On calling this method, it is required that
     *  variables of related unification-nodes have been created.
     *  @return Whether the process succeeded or not. */
    bool add_constraints_of_transitive_unification(
        term_t t1, term_t t2, term_t t3,
        bool do_add_requisite_variable = false);
    void add_constraints_of_transitive_unifications(
        bool do_add_requisite_variable = false);

    void add_constrains_of_conditions_for_chain(pg::edge_idx_t idx);
    void add_constrains_of_exclusive_chains();

    inline void add_constancy_of_variable(variable_idx_t idx, double value);
    inline const hash_map<variable_idx_t, double>& const_variable_values() const;
    inline double const_variable_value(variable_idx_t i) const;
    inline bool is_constant_variable(variable_idx_t) const;

    inline void add_laziness_of_constraint(constraint_idx_t idx);
    inline const hash_set<constraint_idx_t>& get_lazy_constraints() const;

    inline const std::vector<variable_t>& variables() const;
    inline const variable_t& variable(variable_idx_t) const;
    inline       variable_t& variable(variable_idx_t);

    inline const std::vector<constraint_t>& constraints() const;
    inline const constraint_t& constraint(constraint_idx_t) const;
    inline       constraint_t& constraint(constraint_idx_t);

    inline const pg::proof_graph_t* const proof_graph() const;

    /** Return the index of variable corresponding to the given node.
     *  If no variable is found, return -1. */
    inline variable_idx_t find_variable_with_node(pg::node_idx_t) const;

    /** Return the index of variable corresponding to the given hypernode.
     *  If no variable is found, return -1. */
    inline variable_idx_t
        find_variable_with_hypernode(pg::hypernode_idx_t) const;
    template<class T> variable_idx_t
        find_variable_with_hypernode_unordered(T begin, T end) const;
    
    double get_value_of_objective_function(
        const std::vector<double> &values) const;
    
    void print( std::ostream *os ) const;
    std::string to_string() const;

    virtual void print_solution(
        const ilp_solution_t *sol, std::ostream *os) const;

protected:
    virtual void print_node_in_solution(
        pg::node_idx_t i, const ilp_solution_t *sol, std::ostream *os) const;
    virtual void print_chain_in_solution(
        pg::edge_idx_t i, const ilp_solution_t *sol, std::ostream *os) const;
    virtual void print_unification_in_solution(
        pg::edge_idx_t i, const ilp_solution_t *sol, std::ostream *os) const;

    /** A sub-routine of add_constraints_of_exclusiveness_of_chains_from_*.
     *  @return Number of added constraints. */
    size_t add_constrains_of_exclusive_chains(
        const std::list< hash_set<pg::edge_idx_t> > &exc);

    const pg::proof_graph_t* const m_graph;
    
    std::vector<variable_t>   m_variables;
    std::vector<constraint_t> m_constraints;
    double m_cutoff;

    hash_map<variable_idx_t, double> m_const_variable_values;

    /** Indices of constraints which are considered to be lazy
     *  in Cutting Plane Inference. */
    hash_set<constraint_idx_t> m_laziness_of_constraints;
    
    hash_map<pg::node_idx_t, variable_idx_t> m_map_node_to_variable;
    hash_map<pg::hypernode_idx_t, variable_idx_t> m_map_hypernode_to_variable;

    hash_set<size_t> m_hashes_of_term_triplet_for_transitive_unification;
    hash_set<size_t> m_hashes_of_node_tuple_for_mutual_exclusion;
};


enum solution_type_e
{
    SOLUTION_OPTIMAL,
    SOLUTION_SUB_OPTIMAL,
    SOLUTION_NOT_AVAILABLE
};


/** A struct of a solution to a linear-programming-problem. */
class ilp_solution_t
{
public:
    /** Make an instance from current state of given lpp. */
    ilp_solution_t(
        const ilp_problem_t *prob, solution_type_e sol_type,
        const std::vector<double> &values );

    inline solution_type_e type() const;
    inline double value_of_objective_function() const;

    inline bool variable_is_active(variable_idx_t) const;
    inline bool node_is_active(pg::node_idx_t) const;
    inline bool hypernode_is_active(pg::hypernode_idx_t) const;
    inline bool edge_is_active(pg::edge_idx_t) const;
    inline bool constraint_is_satisfied(constraint_idx_t idx) const;

    /** Exclude unsatisfied constraints from targets
     *  and insert them into filtered.
     *  This method is to be used in Cutting Plane algorithm. */
    void filter_unsatisfied_constraints(
        hash_set<constraint_idx_t> *targets,
        hash_set<constraint_idx_t> *filtered) const;

    std::string to_string() const;
    void print(std::ostream *os = &std::cout) const;
    void print_graph(std::ostream *os = &std::cout) const; 
   
private:
    const ilp_problem_t* const m_ilp;
    solution_type_e m_solution_type;
    
    std::vector<double> m_optimized_values;
    std::vector<bool> m_constraints_sufficiency;
    double m_value_of_objective_function;
};


}

}


#include "./ilp_problem.inline.h"


#endif