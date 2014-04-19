#pragma once

#include "../phillip.h"

namespace phil
{

/** A namespace about factories of linear-programming-problems. */
namespace ilp
{


class null_converter_t : public ilp_converter_t
{
public:
    null_converter_t() {}
    virtual ilp::ilp_problem_t* execute() const;
    virtual bool is_available(std::list<std::string>*) const;
    virtual std::string repr() const;
};


/** A class of ilp-converter for a weight-based evaluation function. */
class weighted_converter_t : public ilp_converter_t
{
public:
    struct weight_provider_t {
        virtual ~weight_provider_t() {}
        virtual std::vector<double> operator()(
            const pg::proof_graph_t*, pg::edge_idx_t) const = 0;
    };

    struct fixed_weight_provider_t : public weight_provider_t {
        fixed_weight_provider_t(double w = 1.2) : weight(w) {}
        virtual std::vector<double> operator()(
            const pg::proof_graph_t*, pg::edge_idx_t) const;
        double weight;
    };

    struct basic_weight_provider_t : public weight_provider_t {
        virtual std::vector<double> operator()(
            const pg::proof_graph_t*, pg::edge_idx_t) const;
    };

    weighted_converter_t(
        double default_obs_cost = 10.0,
        weight_provider_t *ptr = NULL);
    ~weighted_converter_t();
    virtual ilp::ilp_problem_t* execute() const;
    virtual bool is_available(std::list<std::string>*) const;
    virtual std::string repr() const;

protected:
    void add_variable_for_cost(
        pg::node_idx_t idx, double cost, ilp::ilp_problem_t *prob,
        hash_map<pg::node_idx_t, ilp::variable_idx_t> *node2costvar) const;
    void add_variables_for_observation_cost(
        const pg::proof_graph_t *graph,
        const lf::input_t &input, ilp::ilp_problem_t *prob,
        hash_map<pg::node_idx_t, ilp::variable_idx_t> *node2costvar) const;
    void add_variables_for_hypothesis_cost(
        const pg::proof_graph_t *graph, ilp::ilp_problem_t *prob,
        hash_map<pg::node_idx_t, ilp::variable_idx_t> *node2costvar) const;
    void add_constraints_for_cost(
        const pg::proof_graph_t *graph, ilp::ilp_problem_t *prob,
        const hash_map<pg::node_idx_t, ilp::variable_idx_t> &node2costvar) const;

    double m_default_observation_cost;
    weight_provider_t *m_weight_provider;
};


/** A class of ilp-converter for a cost-based evaluation function. */
class costed_converter_t : public ilp_converter_t
{
public:
    struct cost_provider_t {
        virtual ~cost_provider_t() {}
        virtual double operator()(
            const pg::proof_graph_t*, pg::edge_idx_t) const = 0;
    };

    struct fixed_cost_provider_t : public cost_provider_t {
        fixed_cost_provider_t(double c = 1.0) : cost(c) {}
        virtual double operator()(
            const pg::proof_graph_t*, pg::edge_idx_t) const;
        double cost;
    };

    struct basic_cost_provider_t : public cost_provider_t {
        virtual double operator()(
            const pg::proof_graph_t*, pg::edge_idx_t) const;
    };

    costed_converter_t(cost_provider_t *ptr = NULL);
    ~costed_converter_t();

    virtual ilp::ilp_problem_t* execute() const;
    virtual bool is_available(std::list<std::string>*) const;
    virtual std::string repr() const;

protected:
    cost_provider_t *m_cost_provider;
};

}

}

