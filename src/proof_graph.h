#pragma once

/** Definition of classes related of proof-graphs.
 *  A proof-graph is used to express a latent-hypotheses-set.
 *  @file   proof_graph.h
 *  @author Kazeto.Y
 */

#include <iostream>
#include <sstream>
#include <string>
#include <set>
#include <map>
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

class node_t;
class edge_t;
class proof_graph_t;


/** An enum of node-type. */
enum node_type_e
{
    NODE_UNDERSPECIFIED, /**< Unknown type. */
    NODE_OBSERVABLE,     /**< The node coressponds an observation. */
    NODE_HYPOTHESIS,     /**< The node corresponds a hypothesis. */
    NODE_REQUIRED           /**< The node corresponds a literal added
                          *   to infer pseudo-positive example. */
};


/** A struct of node in proof-graphs. */
class node_t
{
public:
    /** @param lit     The literal assigned to this.
     *  @param type    The node type of this.
     *  @param idx     The index of this in proof_graph_t::m_nodes.
     *  @param depth   Distance from observations in the proof-graph.
     *  @param parents Indices of nodes being parents of this node. */
    node_t(
        const proof_graph_t *graph,
        const literal_t &lit, node_type_e type, node_idx_t idx,
        depth_t depth, const hash_set<node_idx_t> &parents);

    inline node_type_e type() const { return m_type; }
    inline const literal_t& literal() const { return m_literal; }
    inline arity_t arity() const { return m_literal.get_arity(); }
    inline kb::arity_id_t arity_id() const { return m_arity_id; }

    /** Returns the index of this node in a proof-graph. */
    inline index_t index() const { return m_index; }

    /** Returns the distance from nearest-observation in proof-graph.
     *  Observation and Labels have depth of 0.
     *  Unification-nodes have depth of -1. */
    inline depth_t depth() const { return m_depth; }

    inline const hash_set<pg::node_idx_t>& parents() const;

    /** Returns nodes between this and observations which this explains. */
    inline const hash_set<pg::node_idx_t>& ancestors() const;

    /** Returns nodes which must be hypothesized to hypothesize this. */
    inline const hash_set<pg::node_idx_t>& relatives() const;

    /** Returns the index of hypernode
     *  which was instantiated for instantiation of this node.
     *  CAUTION:
     *    If the node has plural parental-edges, this value is invalid.
     *    Such case can occurs when the node is a substituion node. */
    inline hypernode_idx_t master_hypernode() const;
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
    depth_t m_depth;
    kb::arity_id_t m_arity_id;

    hash_set<node_idx_t> m_parents;
    hash_set<node_idx_t> m_ancestors;
    hash_set<node_idx_t> m_relatives;
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
    inline bool is_unify_edge() const;

private:
    edge_type_e m_type;
    hypernode_idx_t m_index_tail;  /**< A index of tail hypernode. */
    hypernode_idx_t m_index_head;  /**< A index of head hypernode. */
    axiom_id_t m_axiom_id;         /**< The used axiom's id. */
};


/** A class to express unifications of terms.
 *  This class assumes that each unification is one-to-one.
 *  Therefore, for example,
 *  a unifier which has both of (= x y) and (= x z) is invalid. */
class unifier_t
{
public:
    inline unifier_t() {};
    inline unifier_t(const term_t& x, const term_t& y);

    bool operator==(const unifier_t &x) const;

    /** Substitute variables in the given literal.
    *  e.g. (= x y) & p(y) --apply--> p(x) */
    void operator()(literal_t *p_out_lit) const;

    inline void clear();

    /** Add pair of term which are unified. */
    inline void add(term_t x, term_t y);

    inline const std::set<literal_t>& substitutions() const;
    inline const hash_map<term_t, term_t>& mapping() const;

    /** Returns the term which x is substituted to.
     *  If not found, returns NULL. */
    inline const term_t* find_substitution_term(const term_t &x) const;

    /** Returns whether this instance does not have any term. */
    inline bool empty() const;

    bool do_contain(const unifier_t &x) const;

    std::string to_string() const;

private:
    /** The list of unification-assumptions. */
    std::set<literal_t> m_substitutions;

    /** Map from the term before substitution
     *  to the term after substitution. */
    hash_map<term_t, term_t> m_mapping;
};


/** A class of a candidate of chaining.
 *  This class can be used as the key of std::map or the value of std::set. */
struct chain_candidate_t
{
    chain_candidate_t() : axiom_id(0), is_forward(false) {}
    inline chain_candidate_t(
        const std::vector<node_idx_t> &_nodes,
        axiom_id_t _id, bool _is_forward)
        : axiom_id(_id), nodes(_nodes), is_forward(_is_forward) {}

    bool operator>(const chain_candidate_t &x) const;
    bool operator<(const chain_candidate_t &x) const;
    bool operator==(const chain_candidate_t &x) const;
    bool operator!=(const chain_candidate_t &x) const;

    std::vector<node_idx_t> nodes;
    axiom_id_t axiom_id;
    bool is_forward;
};


struct requirement_t
{
    struct element_t
    {
        literal_t literal;
        node_idx_t index;
    };
    std::list<element_t> conjunction;
    bool is_gold;
};


/** A class to express proof-graph of latent-hypotheses-set. */
class proof_graph_t
{
public:
    /** A class to generate candidates of chaining. */
    class chain_candidate_generator_t
    {
    public:
        chain_candidate_generator_t(const proof_graph_t *g);

        void init(node_idx_t);
        void next();
        bool end() const { return m_pt_iter == m_patterns.end(); }
        bool empty() const { return m_axioms.empty(); }

        const std::list<std::vector<node_idx_t> >& targets() const { return m_targets; }
        const std::list<std::pair<axiom_id_t, bool> >& axioms() const { return m_axioms; }

    private:
        void enumerate();

        const proof_graph_t *m_graph;
        node_idx_t m_pivot;

        std::set<kb::arity_pattern_t> m_patterns;
        std::set<kb::arity_pattern_t>::const_iterator m_pt_iter;

        std::list<std::vector<node_idx_t> > m_targets;
        std::list<std::pair<axiom_id_t, bool> > m_axioms;
    };

    /** A class to detect potential loops in a proof-graph. */
    class loop_detector_t
    {
    public:
        loop_detector_t(const proof_graph_t *g);
        const std::list<std::list<edge_idx_t>>& loops() const { return m_loops; }

    private:
        void construct();
        const proof_graph_t *m_graph;
        std::list<std::list<edge_idx_t>> m_loops;
    };

    proof_graph_t(phillip_main_t *main, const std::string &name = "");

    inline phillip_main_t* phillip() const { return m_phillip; }
    inline void timeout(bool flag) { m_is_timeout = flag; }
    inline bool has_timed_out() const { return m_is_timeout; }
    inline const std::string& name() const { return m_name; }

    /** Deletes logs and enumerate hypernodes to be disregarded.
     *  Call this method after creation of proof-graph. */
    void post_process();

    inline node_idx_t add_observation(const literal_t &lit, int depth = 0);

    /** Add an element of requirements.
     *  The operator of req must be OPR_LITERAL or OPR_OR. */
    void add_requirement(const lf::logical_function_t &req);

    /** Add a new hypernode to this proof graph and update maps.
     *  If a hypernode which has same nodes already exists, return its index.
     *  @param indices Ordered node-indices of the new hyper-node.
     *  @return The index of added new hyper-node. */
    hypernode_idx_t add_hypernode(const std::vector<node_idx_t> &indices);

    /** Perform backward-chaining from the target node.
     *  @param axiom The logical function of implication to use.
     *  @return Index of new hypernode resulted in backward-chaining. */
    inline hypernode_idx_t backward_chain(
        const std::vector<node_idx_t> &target, const lf::axiom_t &axiom);

    /** Perform forward-chaining from the target node.
     *  @param axiom The logical function of implication to use.
     *  @return Index of new hypernode resulted in forward-chaining. */
    inline hypernode_idx_t forward_chain(
        const std::vector<node_idx_t> &target, const lf::axiom_t &axiom);

    inline const std::vector<node_t>& nodes() const;
    inline const node_t& node(node_idx_t i) const;

    inline const std::vector<edge_t>& edges() const;
    inline const edge_t& edge(edge_idx_t i) const;

    inline const std::vector< std::vector<node_idx_t> >& hypernodes() const;
    inline const std::vector<node_idx_t>& hypernode(hypernode_idx_t i) const;

    /** Returns a set of indices of observable nodes. */
    inline const hash_set<node_idx_t>& observation_indices() const;

    inline const std::vector<requirement_t>& requirements() const;

    std::list<std::tuple<node_idx_t, node_idx_t, unifier_t> >
        enumerate_mutual_exclusive_nodes() const;

    std::list<hash_set<edge_idx_t> > enumerate_mutual_exclusive_edges() const;

    /** Return pointer of unifier for mutual-exclusiveness between given nodes.
     *  If not found, return NULL. */
    inline const unifier_t* search_mutual_exclusion_of_node(node_idx_t n1, node_idx_t n2) const;

    /** Return pointer of set of nodes whose literal has given term.
     *  If any node was found, return NULL. */
    inline const hash_set<node_idx_t>* search_nodes_with_term(term_t term) const;

    /** Return pointer of set of nodes whose literal has given predicate.
     *  If any node was found, return NULL. */
    inline const hash_set<node_idx_t>* search_nodes_with_predicate(predicate_t predicate, int arity) const;

    /** Return pointer of set of nodes whose literal has given predicate.
     *  If any node was found, return NULL. */
    inline const hash_set<node_idx_t>* search_nodes_with_arity(const arity_t &arity) const;
    inline const hash_set<node_idx_t>* search_nodes_with_arity(kb::arity_id_t arity) const;

    /** Return pointer of set of nodes whose depth is equal to given value.
     *  If any node was found, return NULL. */
    inline const hash_set<node_idx_t>* search_nodes_with_depth(depth_t depth) const;

    /** Return a set of nodes which is unifiable with a literal of given arity.
     *  The threshold of category-table is given
     *  through the parameter "threshold_soft_unify". */
    void enumerate_nodes_softly_unifiable(
        const arity_t &arity, hash_set<node_idx_t> *out) const;

    /** Return set of nodes whose literal is equal to given literal. */
    hash_set<node_idx_t> enumerate_nodes_with_literal(const literal_t &lit) const;

    /** Return the indices of edges connected with given hypernode.
    *  If any edge was not found, return NULL. */
    inline const hash_set<edge_idx_t>*
        search_edges_with_hypernode(hypernode_idx_t idx) const;
    inline const hash_set<edge_idx_t>*
        search_edges_with_node_in_tail(node_idx_t idx) const;
    inline const hash_set<edge_idx_t>*
        search_edges_with_node_in_head(node_idx_t idx) const;

    /** Return the indices of edges which are related with given node. */
    hash_set<edge_idx_t> enumerate_edges_with_node(node_idx_t idx) const;

    /** Return the index of edge connects between
     *  the given hypernode and its parent hypernode.
     *  If any edge is not found, return -1. */
    edge_idx_t find_parental_edge(hypernode_idx_t idx) const;

    /** Return one of parent hypernodes of a given hypernode.
     *  This method is not available to hypernodes for unification,
     *  because they can have plural parent.
     *  If any hypernode was not found, return -1. */
    inline hypernode_idx_t find_parental_hypernode(hypernode_idx_t idx) const;

    void enumerate_parental_edges(hypernode_idx_t idx, hash_set<edge_idx_t> *out) const;
    void enumerate_children_edges(hypernode_idx_t idx, hash_set<edge_idx_t> *out) const;
    void enumerate_parental_hypernodes(hypernode_idx_t idx, hash_set<hypernode_idx_t> *out) const;
    void enumerate_children_hypernodes(hypernode_idx_t idx, hash_set<hypernode_idx_t> *out) const;

    /** Returns indices of nodes whose evidences include given node. */
    void enumerate_descendant_nodes(node_idx_t idx, hash_set<node_idx_t> *out) const;

    void enumerate_overlapping_hypernodes(hypernode_idx_t idx, hash_set<hypernode_idx_t> *out) const;

    /** Return pointer of set of indices of hypernode which has the given node as its element.
     *  If any set was found, return NULL. */
    inline const hash_set<hypernode_idx_t>* search_hypernodes_with_node(node_idx_t i) const;

    /** Return the index of first one of hypernodes whose elements are same as given indices.
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

    /** Returns index of the unifying edge which unifies node i & j. */
    edge_idx_t find_unifying_edge(node_idx_t i, node_idx_t j) const;

    inline const hash_set<term_t>* find_variable_cluster(term_t t) const;
    std::list< const hash_set<term_t>* > enumerate_variable_clusters() const;

    /** Returns a list of chains which are needed to hypothesize given node. */
    hash_set<edge_idx_t> enumerate_dependent_edges(node_idx_t) const;

    /** Returns a list of chains which are needed to hypothesize given node. */
    void enumerate_dependent_edges(node_idx_t, hash_set<edge_idx_t>*) const;

    /** Returns a list of nodes which are needed to hypothesize given node. */
    void enumerate_dependent_nodes(node_idx_t, hash_set<node_idx_t>*) const;

    /** Returns gaps of predicate on given edge.
     *  The first is expected arity and the second is actual arity. */
    std::list<std::pair<arity_t, arity_t> > get_gaps_on_edge(edge_idx_t) const;

    /** Enumerates unification nodes
     *  which are needed to satisfy conditions for given chaining.
     *  @param subs1 Unifying nodes which must be true.
     *  @param subs2 Unifying nodes which must not be true.
     *  @return Whether the chaining is possible. */
    bool check_availability_of_chain(
        pg::edge_idx_t idx,
        hash_set<node_idx_t> *subs1, hash_set<node_idx_t> *subs2) const;

    /** Returns whether nodes in given array can coexist. */
    template <class IterNodesArray>
    bool check_nodes_coexistability(IterNodesArray begin, IterNodesArray end) const;

    std::string hypernode2str(hypernode_idx_t i) const;
    std::string edge_to_string(edge_idx_t i) const;

    /** Returns whether the hypernode of hn includes only sub-nodes. */
    inline bool is_hypernode_for_unification(hypernode_idx_t hn) const;

    /** Returns whether given axioms has already applied to given hypernode. */
    bool axiom_has_applied(
        hypernode_idx_t hn, const lf::axiom_t &ax, bool is_backward) const;

    inline void add_attribute(const std::string &name, const std::string &value);

    inline float threshold_distance_for_soft_unifying() const;

    virtual void print(std::ostream *os) const;

protected:
    /** A class of variable cluster.
     *  Elements of this are terms which are unifiable each other. */
    class unifiable_variable_clusters_set_t
    {
    public:
        unifiable_variable_clusters_set_t() : m_idx_new_cluster(0) {}

        /** Add unifiability of terms t1 & t2. */
        void add(term_t t1, term_t t2);

        void merge(const unifiable_variable_clusters_set_t &vc);

        inline const hash_map<index_t, hash_set<term_t> >& clusters() const;
        inline const hash_set<term_t>* find_cluster(term_t t) const;

        /** Check whether terms t1 & t2 are unifiable. */
        inline bool is_in_same_cluster(term_t t1, term_t t2) const;

    private:
        int m_idx_new_cluster;
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
        const literal_t &p1, const literal_t &p2,
        bool do_ignore_truthment, unifier_t *out = NULL);

    /** Return hash of node indices' list. */
    static size_t get_hash_of_nodes(std::list<node_idx_t> nodes);

    /** Adds a new node and updates maps.
     *  Here, mutual-exclusion and unification-assumptions for the new node
     *  are considered automatically.
     *  @param lit   A literal which the new node has.
     *  @param type  The type of the new node.
     *  @param depth The depth of the new node.
     *  @return The index of added new node. */
    node_idx_t add_node(
        const literal_t &lit, node_type_e type, int depth,
        const hash_set<node_idx_t> &parents);

    /** Adds a new edge.
     *  @return The index of added new edge. */
    edge_idx_t add_edge(const edge_t &edge);

    /** Performs backward-chaining or forward-chaining.
     *  Correspondence of each term is considered on chaining.
     *  @return Index of the new hypernode. If chaining has failed, returns -1. */
    hypernode_idx_t chain(
        const std::vector<node_idx_t> &from,
        const lf::axiom_t &axiom, bool is_backward);

    /** Get mutual exclusions around the literal 'target'. */
    void get_mutual_exclusions(
        const literal_t &target,
        std::list<std::tuple<node_idx_t, unifier_t> > *muexs) const;

    /** Is a sub-routine of add_node.
     *  Generates unification assumptions between target node
     *  and other nodes which have same predicate as target node has. */
    void _generate_unification_assumptions(node_idx_t target);

    /** Is a sub-routine of add_node.
     *  Generates mutual exclusiveness between target node and other nodes.
     *  @param muexs A list of mutual exclusions to create. They are needed to be enumerate by get_mutual_exclusions. */
    void _generate_mutual_exclusions(
        node_idx_t target,
        const std::list<std::tuple<node_idx_t, unifier_t> > &muexs);

    /** Is a sub-routine of _get_mutual_exclusion.
     *  Adds mutual-exclusions for target and nodes being inconsistent with it. */
    void _enumerate_mutual_exclusion_for_inconsistent_nodes(
        const literal_t &target,
        std::list<std::tuple<node_idx_t, unifier_t> > *out) const;

    /** Is a sub-routine of _get_mutual_exclusion.
     *  Adds mutual-exclusions between target and its counter nodes. */
    void _enumerate_mutual_exclusion_for_counter_nodes(
        const literal_t &target,
        std::list<std::tuple<node_idx_t, unifier_t> > *out) const;

    void _enumerate_mutual_exclusion_for_argument_set(
        const literal_t &target,
        std::list<std::tuple<node_idx_t, unifier_t> > *out) const;

    /** Is a sub-routine of chain.
     *  @param is_node_base Gives the mode of enumerating candidate edges.
     *                      If true, enumeration is performed on node-base.
     *                      Otherwise, it is on hypernode-base. */
    void _generate_mutual_exclusion_for_edges(edge_idx_t target, bool is_node_base);

    /** Returns whether given two nodes can coexistence in a hypothesis.
     *  @param uni The pointer of unifier between n1 and n2.
     *  @return Whether given nodes can coexist. */
    bool _check_nodes_coexistability(
        node_idx_t n1, node_idx_t n2, const unifier_t *uni = NULL) const;

    /** This is sub-routine of generate_unification_assumptions.
     *  Add a node and an edge for unification between node[i] & node[j].
     *  And, update m_vc_unifiable and m_maps.terms_to_sub_node. */
    void _chain_for_unification(node_idx_t i, node_idx_t j);

    inline bool _is_considered_unification(node_idx_t i, node_idx_t j) const;

    /** Return highest depth in nodes which given hypernode includes. */
    inline int get_depth_of_deepest_node(hypernode_idx_t idx) const;
    int get_depth_of_deepest_node(const std::vector<node_idx_t> &nodes) const;

    /** If you want to make conditions for unification,
     *  you can override this method. */
    virtual bool can_unify_nodes(node_idx_t, node_idx_t) const { return true; }

    void print_nodes(std::ostream *os) const;
    void print_axioms(std::ostream *os) const;
    void print_edges(std::ostream *os) const;
    void print_subs(std::ostream *os) const;
    void print_mutual_exclusive_nodes(std::ostream *os) const;
    void print_mutual_exclusive_edges(std::ostream *os) const;

    // ---- VARIABLES

    phillip_main_t *m_phillip;

    std::string m_name;
    bool m_is_timeout; /// For timeout.

    std::vector<node_t> m_nodes;
    std::vector< std::vector<node_idx_t> > m_hypernodes;
    std::vector<edge_t> m_edges;

    hash_set<node_idx_t> m_observations; /// Indices of observation nodes.
    std::vector<requirement_t> m_requirements;

    /** These are written in xml-file of output as attributes. */
    hash_map<std::string, std::string> m_attributes;

    float m_threshold_distance_for_soft_unify;

    /** Mutual exclusiveness betwen two nodes.
     *  If unifier of third value is satisfied, the node of the first key and the node of the second key cannot be hypothesized together. */
    util::triangular_matrix_t<node_idx_t, unifier_t> m_mutual_exclusive_nodes;

    hash_map<edge_idx_t, hash_set<edge_idx_t> > m_mutual_exclusive_edges;

    unifiable_variable_clusters_set_t m_vc_unifiable;

    /** Indices of hypernodes which include unification-nodes. */
    hash_set<hypernode_idx_t> m_indices_of_unification_hypernodes;

    /** Substitutions which is needed for the edge of key being true. */
    hash_map<edge_idx_t, std::list< std::pair<term_t, term_t> > > m_subs_of_conditions_for_chain;
    hash_map<edge_idx_t, std::list< std::pair<term_t, term_t> > > m_neqs_of_conditions_for_chain;

    std::hash<std::string> m_hasher_for_nodes;

    struct temporal_variables_t
    {
        void clear();

        /** Set of pair of nodes whose unification was postponed. */
        util::pair_set_t<node_idx_t> postponed_unifications;

        /** Set of pair of nodes
        *  whose unifiability has been already considered.
        *  KEY and VALUE express node pair, and KEY is less than VALUE. */
        util::pair_set_t<node_idx_t> considered_unifications;

        /** Used in _check_nodes_coexistability. */
        mutable util::triangular_matrix_t<node_idx_t, bool> coexistability_logs;

        std::map<std::pair<pg::node_idx_t, term_idx_t>, unsigned long int> argument_set_ids;
    } m_temporal;

    struct maps_t
    {
        /** Map from terms to the node index.
         *   - KEY1, KEY2 : Terms. KEY1 is less than KEY2.
         *   - VALUE : Index of node of "KEY1 == KEY2". */
        util::triangular_matrix_t<term_t, node_idx_t> terms_to_sub_node;

        /** Map from terms to the node index.
         *   - KEY1, KEY2 : Terms. KEY1 is less than KEY2.
         *   - VALUE : Index of node of "KEY1 != KEY2". */
        util::triangular_matrix_t<term_t, node_idx_t> terms_to_negsub_node;

        /** Map from depth to indices of nodes assigned the depth. */
        hash_map<depth_t, hash_set<node_idx_t> > depth_to_nodes;

        /** Map from axiom-id to hypernodes which have been applied the axiom. */
        hash_map< axiom_id_t, hash_set<hypernode_idx_t> >
            axiom_to_hypernodes_forward, axiom_to_hypernodes_backward;

        /** Map to get node from predicate.
         *  This is used on enumerating unification assumputions.
         *   - KEY1  : Predicate of the literal.
         *   - KEY2  : Num of terms of the literal.
         *   - VALUE : Indices of nodes which have the corresponding literal. */
        hash_map<predicate_t, hash_map<int, hash_set<node_idx_t> > >
            predicate_to_nodes;

        /** Map to get hypernodes which include given node. */
        hash_map<node_idx_t, hash_set<hypernode_idx_t> > node_to_hypernode;

        /** Map to get hypernodes from hash of unordered-nodes. */
        hash_map<size_t, hash_set<hypernode_idx_t> > unordered_nodes_to_hypernode;

        /** Map to get edges connecting given node. */
        hash_map<hypernode_idx_t, hash_set<edge_idx_t> > hypernode_to_edge;

        hash_map<node_idx_t, hash_set<edge_idx_t> > tail_node_to_edges, head_node_to_edges;

        /** Map to get nodes which have given term. */
        hash_map<term_t, hash_set<node_idx_t> > term_to_nodes;

        hash_map<kb::arity_id_t, hash_set<node_idx_t> > arity_to_nodes;
        hash_map<kb::arity_id_t, hash_set<node_idx_t> > arity_wc_to_nodes;
    } m_maps;
};


}

}

#include "proof_graph.inline.h"
