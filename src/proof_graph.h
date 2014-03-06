/* -*- coding: utf-8 -*- */

#ifndef INCLUDE_HENRY_PROOF_GRAPH_H
#define INCLUDE_HENRY_PROOF_GRAPH_H

/** Definition of classes related of proof-graphs.
 *  A proof-graph is used to express a latent-hypotheses-set.
 *  @file   proof_graph.h
 *  @author Kazeto.Y
 */

#include <iostream>
#include <sstream>
#include <string>
#include <ciso646>


#include "./logical_function.h"


namespace phil
{


namespace ilp
{
class ilp_problem_t;
class ilp_solution_t;
}


/** A namespace about proof-graphs. */
namespace pg
{

typedef index_t entity_idx_t;
typedef index_t node_idx_t;
typedef index_t edge_idx_t;
typedef index_t hypernode_idx_t;

typedef hash_map<int, std::vector<hypernode_idx_t> > node_hypernode_map_t;
typedef hash_map<size_t, std::vector<index_t> > eqhash_t;
typedef hash_map<index_t, std::vector<index_t> > edge_set_t;
typedef hash_map<term_t, std::vector<index_t> > term_map_t;
typedef hash_map<term_t, hash_map<index_t, std::vector<index_t> > > node_map_t;
typedef hash_map<index_t, hash_set<term_t> > cluster_t;
typedef hash_map<term_t, index_t>            variable_mapper_t;

class node_t;
class edge_t;
class proof_graph_t;


/** An enum of node-type. */
enum node_type_e
{
    NODE_UNDERSPECIFIED, /**< Unknown type. */
    NODE_OBSERVABLE,     /**< The node coressponds an observation. */
    NODE_HYPOTHESIS,     /**< The node corresponds a hypothesis. */
    NODE_LABEL           /**< The node corresponds a literal added
                          *   to infer pseudo-positive example. */
};


/** A struct of node in proof-graphs. */
class node_t
{
public:
    /** @param lit   The literal assigned to this.
     *  @param type  The node type of this.
     *  @param idx   The index of this in proof_graph_t::m_nodes.
     *  @param depth Distance from observations in the proof-graph. */
    inline node_t(
        const literal_t &lit, node_type_e type, node_idx_t idx, int depth);

    inline node_type_e type() const         { return m_type; }
    inline const literal_t& literal() const { return m_literal; }

    /** Return the index of this node in a proof-graph. */
    inline index_t index() const { return m_index; }

    /** Return the distance from nearest-observation in proof-graph.
     *  Observation and Labels have depth of 0.
     *  Unification-nodes have depth of -1. */
    inline int depth() const { return m_depth; }

    /** List of inconsistency of variables.
    *  If these are violated, this node cannot be hypothesized. */
    inline const std::vector< std::pair<term_t, term_t> >&
        get_conditions_for_non_equality_of_terms() const;

    /** Return the index of hypernode
     *  which was instantiated for instantiation of this node.
     *  CAUTION:
     *    If the node has plural parental-edges, this value is invalid.
     *    Such case can occurs when the node is a substituion node. */
    inline hypernode_idx_t get_master_hypernode() const;
    inline void set_master_hypernode(hypernode_idx_t idx);

    /** If true, this node is a substitution node. */
    inline bool is_equality_node() const;

    /** If true, this node is a non-equality node. */
    inline bool is_non_equality_node() const;

    /** If true, this node can be hypothesized
     *  by only transitive unification from other nodes. */
    inline bool is_transitive_equality_node() const;

    inline std::string to_string() const;

private:
    node_type_e m_type;
    literal_t   m_literal;
    node_idx_t  m_index;
    hypernode_idx_t m_master_hypernode_idx;
    int m_depth;

    /** IDs of axioms which has been applied to this node. */
    hash_set<axiom_id_t> m_ids_axiom_used_forward, m_ids_axiom_used_backward;

    std::vector< std::pair<term_t, term_t> > m_conditions_neqs;
};


enum edge_type_e
{
    EDGE_UNDERSPECIFIED,
    EDGE_HYPOTHESIZE, /// For abduction.
    EDGE_IMPLICATION, /// For deduction.
    EDGE_UNIFICATION, /// For unification.
    EDGE_USER_DEFINED
};


/** A struct of edge to express explaining in proof-graphs.
 *  So, in abduction, the direction of implication is opposite.
 *  If this is unification edge, then id_axiom = -1. */
class edge_t
{
public:
    inline edge_t();
    inline edge_t(
        edge_type_e type, hypernode_idx_t tail, hypernode_idx_t head,
        axiom_id_t id = -1 );

    inline edge_type_e type()     const { return m_type; }
    inline hypernode_idx_t tail() const { return m_index_tail; }
    inline hypernode_idx_t head() const { return m_index_head; }
    inline axiom_id_t axiom_id()  const { return m_axiom_id; }

    inline bool is_chain_edge() const;

private:
    edge_type_e m_type;
    hypernode_idx_t m_index_tail;  /**< A index of tail hypernode. */
    hypernode_idx_t m_index_head;  /**< A index of head hypernode. */
    axiom_id_t m_axiom_id;         /**< The used axiom's id. */
};


/** A class to handle unifications of terms.
 *  This class assumes that each unification is one-to-one.
 *  Namely, (x = y) and (x = z) at same time is invalid. */
class unifier_t
{
public:
    inline unifier_t() {};

    /** @param x,y Terms unified. */
    inline unifier_t(const term_t& x, const term_t& y);

    inline void clear();

    /** @return The term which is unified with x. */
    inline const term_t& map(const term_t &x) const;

    inline const std::vector<literal_t>& substitutions() const;

    /** Substitute variables in the given literal.
     *  e.g. (x = y) & p(x) --apply--> p(y) */
    inline void apply(literal_t *p_out_lit) const;

    inline void add(const term_t &x, const term_t &y);
    inline void add(const term_t &x, const std::string &var);

    inline bool has_applied( const term_t &x ) const;
    std::string to_string() const;

private:
    /** The list of unification-assumptions. */
    std::vector<literal_t> m_substitutions;

    /** Map from a term to the index of substitution which includes it. */
    hash_map<term_t, index_t> m_shortcuts;

    /** Map from a term to the term unified with it. */
    hash_map<term_t, term_t> m_mapping;
};


/** A struct to express mutual exclusion among nodes. */
struct mutual_exclusion_t
{
    /** Pair of indices of the nodes. */
    std::pair<node_idx_t, node_idx_t> indices;
    /** A hypothesis is inconsistent if this unifier is satisfied.
    *  Namely, m_substitutions in this unifier must not be satisfied. */
    unifier_t  unifier;
};


/** A class to express proof-graph of latent-hypotheses-set. */
class proof_graph_t
{        
public:
    inline proof_graph_t() {}
    
    inline node_idx_t add_observation(const literal_t &lit, int depth = 0);
    inline node_idx_t add_label(const literal_t &lit, int depth = 0);

    /** Add a new hypernode to this proof graph and update maps.
     *  If a hypernode which has same nodes already exists, return its index.
     *  @param indices Ordered node-indices of the new hyper-node.
     *  @return The index of added new hyper-node. */
    hypernode_idx_t add_hypernode(const std::vector<node_idx_t> &indices);

    /** Perform backward-chaining from the target node.
     *  @param axiom The logical function of implication to use.
     *  @return Index of new hypernode resulted in backward-chaining. */
    inline hypernode_idx_t backward_chain(
        hypernode_idx_t target, const lf::axiom_t &axiom);
    
    /** Perform forward-chaining from the target node.
     *  @param axiom The logical function of implication to use.
     *  @return Index of new hypernode resulted in forward-chaining. */
    inline hypernode_idx_t forward_chain(
        hypernode_idx_t target, const lf::axiom_t &axiom);

    void generate_unification_assumptions(node_idx_t target);
    void generate_mutual_exclusions(node_idx_t target);

    inline const std::vector<node_t>& nodes() const;
    inline const node_t& node(node_idx_t i) const;

    inline const std::vector<edge_t>& edges() const;
    inline const edge_t& edge(edge_idx_t i) const;

    inline const std::vector< std::vector<node_idx_t> >& hypernodes() const;
    inline const std::vector<node_idx_t>& hypernode(hypernode_idx_t i) const;

    inline const std::list<mutual_exclusion_t>& mutual_exclusions() const;

    /** Return pointer of set of nodes whose literal has given term.
     *  If any node was found, return NULL. */
    inline const hash_set<node_idx_t>* search_nodes_with_term(term_t term) const;

    /** Return pointer of set of nodes whose literal has given predicate.
     *  If any node was found, return NULL. */
    inline const hash_set<node_idx_t>*
        search_nodes_with_predicate(predicate_t predicate, int arity) const;

    /** Return pointer of set of nodes whose literal has given predicate.
     *  If any node was found, return NULL. */
    inline const hash_set<node_idx_t>*
        search_nodes_with_arity(std::string arity) const;

    /** Return pointer of set of nodes whose depth is equal to given value.
     *  If any node was found, return NULL. */
    inline const hash_set<node_idx_t>*
        search_nodes_with_depth(int depth) const;

    /** Return set of nodes whose literal is equal to given literal. */
    hash_set<node_idx_t> enumerate_nodes_with_literal(const literal_t &lit) const;

    /** Return the indices of edges connected with given hypernode.
    *  If any edge was not found, return NULL. */
    inline const hash_set<edge_idx_t>*
        search_edges_with_hypernode(hypernode_idx_t idx) const;

    /** Return the index of edge connects between
     *  the given hypernode and its parent hypernode.
     *  If any edge is not found, return -1. */
    edge_idx_t find_parental_edge(hypernode_idx_t idx) const;

    /** Return one of parent hypernodes of a given hypernode.
     *  This method is not available to hypernodes for unification,
     *  because they can have plural parent.
     *  If any hypernode was not found, return -1. */
    inline hypernode_idx_t find_parental_hypernode(hypernode_idx_t idx) const;

    void enumerate_parental_edges(
        hypernode_idx_t idx, std::list<edge_idx_t> *out) const;
    void enumerate_children_edges(
        hypernode_idx_t idx, std::list<edge_idx_t> *out) const;
    void enumerate_hypernodes_children(
        hypernode_idx_t idx, std::list<hypernode_idx_t> *out) const;
    void enumerate_hypernodes_parents(
        hypernode_idx_t idx, std::list<hypernode_idx_t> *out) const;

    /** Enumerate lists of node indices which given axiom is applicable to.
     *  You can use this method to enumerate hypernodes for chaining.
     *  Nodes whose depth exceed max_depth are excluded from enumeration. */
    std::list< std::vector<node_idx_t> > enumerate_targets_of_chain(
        const lf::axiom_t &ax, bool is_backward, int max_depth = -1) const;
    
    /** Enumerate arrays of node indices which corresponds arities.
     *  You can use this method to enumerate hypernodes for chaining.
     *  Nodes whose depth exceed depth_limit are exclude from target. */
    std::list< std::vector<node_idx_t> > enumerate_nodes_list_with_arities(
        const std::vector<std::string> &arities, int depth_limit) const;

    /** Return pointer of set of indices of hypernode
     *  which has the given node as its element.
     *  If any set was found, return NULL. */
    inline const hash_set<hypernode_idx_t>*
        search_hypernodes_with_node(node_idx_t i) const;

    /** Return the index of first one of hypernodes
     *  whose elements are same as given indices.
     *  If any hypernode was not found, return -1.  */
    template<class It> const hash_set<hypernode_idx_t>*
        find_hypernode_with_unordered_nodes(It begin, It end) const;

    /** Return the index of hypernode whose elements are same as given indices.
     *  If any hypernode was not found, return -1.  */
    hypernode_idx_t find_hypernode_with_ordered_nodes(
        const std::vector<node_idx_t> &indices) const;

    node_idx_t find_sub_node(term_t t1, term_t t2) const;
    node_idx_t find_neg_sub_node(term_t t1, term_t t2) const;

    /** Return index of the substitution nodes which hypothesized by transitivity */
    node_idx_t find_transitive_sub_node(node_idx_t i, node_idx_t j) const;

    /** Insert sub-nodes being transitive to target into target. */
    void insert_transitive_sub_node(hash_set<node_idx_t> *target) const;

    inline const hash_set<term_t>* find_variable_cluster(term_t t) const;
    std::list< const hash_set<term_t>* > enumerate_variable_clusters() const;

    /** Get list of edges in paths from given node to its evidences. */
    std::list<edge_idx_t> enumerate_edges_to_observation(node_idx_t from) const;
    void enumerate_edges_to_observation(
        node_idx_t from, std::list<edge_idx_t> *out) const;

    /** Get list of chains which has same group's axioms in this proof-graph.
     *  Each list of indices is sorted in ascending-order.
     *  This method is used in adding mutual-exclusions of inconsistency. */
    void enumerate_chains_of_grouped_axioms_from_node(
        node_idx_t from, std::list< std::list<edge_idx_t> > *out) const;
    void enumerate_chains_of_grouped_axioms_from_hypernode(
        hypernode_idx_t from, std::list< std::list<edge_idx_t> > *out) const;

    /** Enumerate sub-nodes which are needed to perform the chaining
     *  and return whether the chaining is possible. */
    bool check_availability_of_chain(
        pg::edge_idx_t idx, hash_set<node_idx_t> *out) const;

    std::string edge_to_string(edge_idx_t i) const;
        
    inline bool is_hypernode_for_unification(hypernode_idx_t hn) const;

    bool axiom_has_applied(
        hypernode_idx_t hn, const lf::axiom_t &ax, bool is_backward) const;


    virtual void print(std::ostream *os) const;

protected:
    /** A class of variable cluster.
     *  Elements of this are terms which are unifiable each other. */
    class unifiable_variable_clusters_set_t
    {
    public:
        /** Add unifiability of terms t1 & t2. */
        void add( term_t t1, term_t t2 );

        inline const hash_map<index_t, hash_set<term_t> >& clusters() const;
        inline const hash_set<term_t>* find_cluster(term_t t) const;
        
        /** Check whether terms t1 & t2 are unifiable. */
        inline bool is_in_same_cluster(term_t t1, term_t t2) const;
        
    private:
        /** List of clusters.
         *  We use hash-map for erasure with keeping indices. */
        hash_map<index_t, hash_set<term_t> > m_clusters;

        /** Mapping from name of a variable
         *  to the index of cluster which the variable joins. */
        hash_map<term_t, index_t> m_map_v2c;

        /** All variables included in this cluster-set. */
        hash_set<term_t> m_variables;
    };

    /** Get whether it is possible to unify literals p1 and p2.
     *  @param[in]  p1,p2 Target literals of unification.
     *  @param[out] out   The unifier of p1 and p2.
     *  @return Possibility to unify literals p1 & p2. */
    static bool check_unifiability(
        const literal_t &p1, const literal_t &p2, bool do_ignore_truthment,
        unifier_t *out);

    /** Return hash of node indices' list. */
    static size_t get_hash_of_nodes(std::list<node_idx_t> nodes);

    /** Add a new node and update maps.
     *  @return The index of added new node. */
    node_idx_t add_node(const literal_t &lit, node_type_e type, int depth);

    /** Add a new edge.
     *  @return The index of added new edge. */
    inline edge_idx_t add_edge( const edge_t &edge );

    /** Perform backward-chaining or forward-chaining.
     *  Correspondence of each term is considered on chaining. */
    hypernode_idx_t chain(
        hypernode_idx_t from, const lf::axiom_t &axiom, bool is_backward);

    /* Sub-routines of chain. */
    void get_substitutions_for_chain(
        const std::vector<node_idx_t> &nodes,
        const std::vector<const literal_t*> &li_from,
        hash_map<term_t, term_t> *subs,
        hash_map<term_t, hash_set<term_t> > *conds) const;
    void get_substitutions_for_chain_sub(
        term_t t_from, term_t t_to,
        hash_map<term_t, term_t> *subs,
        hash_map<term_t, hash_set<term_t> > *conds) const;
    term_t substitute_term_for_chain(
        const term_t &target, hash_map<term_t, term_t> *subs) const;

    void _generate_mutual_exclusion_for_inconsistency(node_idx_t target);
    void _generate_mutual_exclusion_for_counter_nodes(node_idx_t target);

    /** Sub-routine of generate_unification_assumptions().
     *  Return indices of node which is unifiable with target.
     *  Node pairs which has been considered once are ignored. */
    std::list<node_idx_t> enumerate_unifiable_nodes(node_idx_t target);

    /** This is sub-routine of generate_unification_assumptions.
     *  Add a node and an edge for unification between node[i] & node[j].
     *  And, update m_vc_unifiable and m_maps.nodes_sub. */
    void _chain_for_unification(node_idx_t i, node_idx_t j);

    /** Sub-routine of chain_for_unification().
     *  Add nodes for transitive unification around the given term. */
    void _add_nodes_of_transitive_unification(term_t t);

    /** Apply inconsistency between i-th node and j-th node. */
    void apply_inconsistency_sub(
        node_idx_t i, node_idx_t j, const lf::axiom_t &axiom);

    inline bool _is_considered_unification(node_idx_t i, node_idx_t j) const;
    inline bool _is_considered_exclusion(node_idx_t i, node_idx_t j) const;

    /** Return highest depth in nodes included in the given hypernode. */
    int get_depth_of_deepest_node(hypernode_idx_t idx) const;

    /** If you want to make conditions for unification,
     *  you can override this method. */
    virtual bool can_unify_nodes(node_idx_t, node_idx_t) const { return true; }

    virtual void print_nodes(
        std::ostream *os, const std::string &indent) const;
    virtual void print_axioms(
        std::ostream *os, const std::string &indent) const;
    virtual void print_edges(
        std::ostream *os, const std::string &indent) const;
    virtual void print_subs(
        std::ostream *os, const std::string &indent) const;
    virtual void print_exclusiveness(
        std::ostream *os, const std::string &indent) const;

    // ---- VARIABLES
    
    std::vector<node_t> m_nodes;
    std::vector< std::vector<node_idx_t> > m_hypernodes;
    std::vector<edge_t> m_edges;
    
    /** List of index of node which is label. */
    std::vector<node_idx_t> m_label_nodes;
    
    std::list<mutual_exclusion_t> m_mutual_exclusions;
    
    unifiable_variable_clusters_set_t m_vc_unifiable;

    /** Set of pair of nodes
     *  which its unifiability has been already considered.
     *  KEY and VALUE express node pair, and KEY is less than VALUE. */
    hash_map<node_idx_t, hash_set<node_idx_t> >
        m_unification_assumptions_considered;
    
    /** Indices of hypernodes which include unification-nodes. */
    hash_set<hypernode_idx_t> m_indices_of_unification_hypernodes;

    /** Substitutions which is needed for the edge of key being true. */
    hash_map<edge_idx_t, std::list< std::pair<term_t, term_t> > >
        m_subs_of_conditions_for_chain;

    struct
    {
        hash_map<node_idx_t, hash_set<node_idx_t> > considered_unifications;
        hash_map<node_idx_t, hash_set<node_idx_t> > considered_exclusions;
    } m_logs;

    struct
    {
        /** Map from terms to the node index.
         *   - KEY1, KEY2 : Terms. KEY1 is less than KEY2.
         *   - VALUE : Index of node of "KEY1 == KEY2". */
        hash_map<term_t, hash_map<term_t, node_idx_t> > nodes_sub;

        /** Map from terms to the node index.
         *   - KEY1, KEY2 : Terms. KEY1 is less than KEY2.
         *   - VALUE : Index of node of "KEY1 != KEY2". */
        hash_map<term_t, hash_map<term_t, node_idx_t> > nodes_negsub;

        /** Map from a variable to constants
         *  which are unified with the variable. */
        hash_map<term_t, hash_set<term_t> > var_to_consts;
        // ---- constants_sub

        /** Map from a node index
         *  to the set of inconsistency axioms related to it. */
        hash_map<node_idx_t, hash_set<axiom_id_t> > node_to_inconsistency;

        /** Map from depth to indices of nodes assigned the depth. */
        hash_map<int, hash_set<node_idx_t> > depth_to_nodes;

        /** Map from axiom-id
         *  to indices of hypernode which has been applied the axiom. */
        hash_map< axiom_id_t, hash_set<hypernode_idx_t> >
            axiom_to_hypernodes_forward, axiom_to_hypernodes_backward;

        /** Map to get node from predicate.
         *  This is used on enumerating unification assumputions.
         *   - KEY1  : Predicate of the literal.
         *   - KEY2  : Num of terms of the literal.
         *   - VALUE : Indices of nodes which have the corresponding literal. */
        hash_map<predicate_t, hash_map<int, hash_set<node_idx_t> > >
            predicate_to_nodes;

        /** Map to get hypernodes from node.
         *   - KEY : Index of node.
         *   - VALUE : Index of hypernode which includes KEY. */
        hash_map<node_idx_t, hash_set<hypernode_idx_t> > node_to_hypernode;

        /** Map to get hypernodes from hash of unordered-nodes. */
        hash_map<size_t, hash_set<hypernode_idx_t> > unordered_nodes_to_hypernode;

        /** Map to get edges connecting some node.
         *   - KEY : Index of node.
         *   - VALUE : Index of edges which connect KEY. */
        hash_map< hypernode_idx_t, hash_set<edge_idx_t> > hypernode_to_edge;

        /** Map to get node from term.
         *   - KEY : Name of term.
         *   - VALUE : Indices of nodes which have the term corresponding KEY. */
        hash_map<term_t, hash_set<node_idx_t> > term_to_nodes;
    } m_maps;
};


}

}

#include "proof_graph.inline.h"

#endif
