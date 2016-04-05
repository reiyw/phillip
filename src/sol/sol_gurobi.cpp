#include <mutex>
#include <algorithm>
#include "./ilp_solver.h"


namespace phil
{

namespace sol
{


#define GRBEXECUTE(x) \
    try { x; } \
    catch (GRBException e) { \
        util::print_error_fmt("Gurobi: code(%d): %s", \
            e.getErrorCode(), e.getMessage().c_str()); }


std::mutex g_mutex_gurobi;


void _add_GRBconstraint(GRBModel *model, const ilp::constraint_t &c, const hash_map<ilp::variable_idx_t, GRBVar> &vars) {
  std::string name = c.name().substr(0, 32);
  GRBLinExpr expr;

  for (auto t = c.terms().begin(); t != c.terms().end(); ++t)
      expr += t->coefficient * vars.at(t->var_idx);

  GRBEXECUTE(
      switch (c.operator_type())
      {
      case ilp::OPR_EQUAL:
          model->addConstr(expr, GRB_EQUAL, c.bound(), name);
          break;
      case ilp::OPR_LESS_EQ:
          model->addConstr(expr, GRB_LESS_EQUAL, c.upper_bound(), name);
          break;
      case ilp::OPR_GREATER_EQ:
          model->addConstr(expr, GRB_GREATER_EQUAL, c.lower_bound(), name);
          break;
      case ilp::OPR_RANGE:
          model->addRange(expr, c.lower_bound(), c.upper_bound(), name);
          break;
      });
}


gurobi_t::gurobi_t(phillip_main_t *ptr, int thread_num, bool do_output_log)
    : ilp_solver_t(ptr), m_thread_num(thread_num), m_do_output_log(do_output_log)
{
    if (m_thread_num <= 0)
        m_thread_num = 1;
}


ilp_solver_t* gurobi_t::duplicate(phillip_main_t *ptr) const
{
    return new gurobi_t(ptr, m_thread_num, m_do_output_log);
}


void gurobi_t::execute(std::vector<ilp::ilp_solution_t> *out) const
{
#ifdef USE_GUROBI
    const ilp::ilp_problem_t *prob = phillip()->get_ilp_problem();
    solve(prob, out);
#endif
}


void gurobi_t::solve(
    const ilp::ilp_problem_t *prob,
    std::vector<ilp::ilp_solution_t> *out) const
{
#ifdef USE_GUROBI
    auto begin = std::chrono::system_clock::now();
    auto get_timeout = [&begin, this]() -> double
    {
        duration_time_t passed = util::duration_time(begin);
        double t_o_sol(-1), t_o_all(-1);

        if (phillip() != NULL)
        {
            if (not phillip()->timeout_sol().empty())
                t_o_sol = std::max<double>(
                0.01,
                phillip()->timeout_sol().get() - passed);
            if (not phillip()->timeout_all().empty())
                t_o_all = std::max<double>(
                0.01,
                phillip()->timeout_all().get()
                - phillip()->get_time_for_lhs()
                - phillip()->get_time_for_ilp()
                - passed);
        }

        double timeout(-1);
        if (t_o_sol > t_o_all)
            timeout = (t_o_all > 0.0) ? t_o_all : t_o_sol;
        else
            timeout = (t_o_sol > 0.0) ? t_o_sol : t_o_all;

        return (timeout > 0.0) ? timeout : -1.0;
    };

    g_mutex_gurobi.lock();
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);
    hash_map<ilp::variable_idx_t, GRBVar> vars;
    hash_set<ilp::constraint_idx_t>
        lazy_cons(prob->get_lazy_constraints());
    g_mutex_gurobi.unlock();

    bool do_cpi(not lazy_cons.empty());
    if (phillip() != NULL)
    if (phillip()->flag("disable_cpi"))
        do_cpi = false;

    add_variables(prob, &model, &vars);

    for (int i = 0; i < prob->constraints().size(); ++i)
    if (lazy_cons.count(i) == 0 or not do_cpi)
        add_constraint(prob, &model, i, vars);

    double timeout = get_timeout();
    GRBEXECUTE(
        model.update();
        model.set(
            GRB_IntAttr_ModelSense,
            (prob->do_maximize() ? GRB_MAXIMIZE : GRB_MINIMIZE));
        model.getEnv().set(GRB_IntParam_OutputFlag, m_do_output_log ? 1 : 0);
        if (m_thread_num > 1)
            model.getEnv().set(GRB_IntParam_Threads, m_thread_num);
        if (timeout > 0)
            model.getEnv().set(GRB_DoubleParam_TimeLimit, timeout););

    int len = 1;
    bool do_kbest(false),
      do_kbest_litwise(false),
      do_kbest_unified_with_constant(false);

    if(phillip() != NULL) {
      do_kbest = phillip()->flag("kbest");

      if(do_kbest) {
        len = phillip()->param_int("kbest_k", 2);
        do_kbest_litwise = phillip()->flag("kbest_litwise");
        do_kbest_unified_with_constant = phillip()->flag("kbest_prohibit_unification_with_constant_only");
      }
    }

    for(int K=0; K<len; K++) {

      size_t num_loop(0);
      while (true)
      {
          if (do_cpi)
              util::print_console_fmt("begin: Cutting-Plane-Inference #%d", (num_loop++));

          GRBEXECUTE(model.optimize());

          if (model.get(GRB_IntAttr_SolCount) == 0)
          {
              if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE)
              {
                  model.computeIIS();
                  GRBConstr *cons = model.getConstrs();

                  for (int i = 0; i < model.get(GRB_IntAttr_NumConstrs); ++i)
                  if (cons[i].get(GRB_IntAttr_IISConstr) == 1)
                  {
                      std::string name(cons[i].get(GRB_StringAttr_ConstrName));
                      util::print_warning("Infeasible: " + name);
                  }

                  delete[] cons;
              }

              if(do_kbest && K>0) break;

              ilp::ilp_solution_t sol(
                  prob, ilp::SOLUTION_NOT_AVAILABLE,
                  std::vector<double>(prob->variables().size(), 0.0));
              out->push_back(sol);
              break;
          }
          else
          {
              ilp::ilp_solution_t sol = convert(prob, &model, vars, prob->name());
              bool do_break(false);
              bool do_violate_lazy_constraint(false);

              if (not lazy_cons.empty() and do_cpi)
              {
                  hash_set<ilp::constraint_idx_t> filtered;
                  sol.filter_unsatisfied_constraints(&lazy_cons, &filtered);

                  if (not filtered.empty())
                  {
                      // ADD VIOLATED CONSTRAINTS
                      for (auto it = filtered.begin(); it != filtered.end(); ++it)
                          add_constraint(prob, &model, *it, vars);
                      model.update();
                      do_violate_lazy_constraint = true;
                  }
                  else do_break = true;
              }
              else do_break = true;

              if (not do_break and phillip() != NULL)
              {
                  if (do_time_out(begin))
                  {
                      sol.timeout(true);
                      do_break = true;
                  }
                  else
                  {
                      double t_o = get_timeout();
                      if (t_o > 0.0)
                          GRBEXECUTE(model.getEnv().set(GRB_DoubleParam_TimeLimit, t_o););
                  }
              }

              if (do_break)
              {
                  bool timeout_lhs =
                      (prob->proof_graph() != NULL) ?
                      prob->proof_graph()->has_timed_out() : false;
                  ilp::solution_type_e sol_type =
                      infer_solution_type(timeout_lhs, prob->has_timed_out(), false);
                  if (do_violate_lazy_constraint)
                      sol_type = ilp::SOLUTION_NOT_AVAILABLE;

                  sol.set_solution_type(sol_type);
                  out->push_back(sol);
                  break;
              }
          }
      }

      if(do_kbest) {
        if(model.get(GRB_IntAttr_SolCount) == 0) {
          util::print_console_fmt("K-BEST: Terminated.");
          break;
        }

        ilp::ilp_solution_t last_sol = (*out)[out->size()-1];

        util::print_console_fmt("K-BEST: Got a solution (obj. = %f)", model.get(GRB_DoubleAttr_ObjVal));

        // Get indices of ILP variables to construct an ILP constraint.
        const pg::proof_graph_t *pg = prob->proof_graph();
        ilp::constraint_t con_suppress("SUPPRESSOR", ilp::OPR_LESS_EQ, 1);
        string strLiterals = "";

        // Find prohibited nodes.
        for(int i=0; i<pg->nodes().size(); i++) {
          if(!prob->node_is_active(last_sol, i)) continue;
          if(-1 == pg->node(i).literal().predicate.find(phillip()->param("kbest_pred"))) continue;

          strLiterals += pg->node(i).to_string() + " ";

          // Furthermore, find active equalities related to the arguments in active nodes.
          for(int j=0; j<pg->node(i).literal().terms.size(); j++) {
            if(pg->node(i).literal().terms[j].is_constant()) continue;

            const hash_set<pg::node_idx_t> *pNodes = pg->search_nodes_with_term(pg->node(i).literal().terms[j]);

            // pNodes: related (possibly non-equality) literals.
            for(auto eq = pNodes->begin(); eq != pNodes->end(); ++eq) {
              if(prob->node_is_active(last_sol, *eq) && (pg->node(*eq).is_equality_node() || pg->node(*eq).is_transitive_equality_node())) {

                // Sometimes it concerns only unification with constants.
                if(do_kbest_unified_with_constant) {
                  if(!pg->node(*eq).literal().terms[0].is_constant() &&
                    !pg->node(*eq).literal().terms[1].is_constant())
                    continue;
                }

                strLiterals += pg->node(*eq).to_string() + " ";
                con_suppress.add_term(prob->find_variable_with_node(*eq), 1.0);
              }
            }
          }

          con_suppress.add_term(prob->find_variable_with_node(i), 1.0);

          if(do_kbest_litwise) {
            con_suppress.set_bound(con_suppress.terms().size() - 1.0);            _add_GRBconstraint(&model, con_suppress, vars);

            con_suppress = ilp::constraint_t("SUPPRESSOR", ilp::OPR_LESS_EQ, 1);
          }
        }

        util::print_console_fmt("K-BEST: To be suppressed: %s",strLiterals.c_str());

        if(!do_kbest_litwise) {
          con_suppress.set_bound(con_suppress.terms().size() - 1.0);
          _add_GRBconstraint(&model, con_suppress, vars);
        }

      } else break;

    }

#endif
}


bool gurobi_t::is_available(std::list<std::string> *err) const
{
#ifdef USE_GUROBI
    return true;
#else
    err->push_back("This binary cannot use gurobi-optimizer.");
    return false;
#endif
}


#ifdef USE_GUROBI

void gurobi_t::add_variables(
    const ilp::ilp_problem_t *prob,
    GRBModel *model, hash_map<ilp::variable_idx_t, GRBVar> *vars) const
{
    for (int i = 0; i < prob->variables().size(); ++i)
    {
        const ilp::variable_t &v = prob->variable(i);
        double lb(0.0), ub(1.0);

        if (prob->is_constant_variable(i))
            lb = ub = prob->const_variable_value(i);

        GRBEXECUTE(
            (*vars)[i] = model->addVar(
                lb, ub, v.objective_coefficient(),
                (ub - lb == 1.0) ? GRB_BINARY : GRB_INTEGER))
    }

    GRBEXECUTE(model->update())
}


void gurobi_t::add_constraint(
    const ilp::ilp_problem_t *prob,
    GRBModel *model, ilp::constraint_idx_t idx,
    const hash_map<ilp::variable_idx_t, GRBVar> &vars) const
{
    const ilp::constraint_t &c = prob->constraint(idx);
    _add_GRBconstraint(model, c, vars);
}


ilp::ilp_solution_t gurobi_t::convert(
    const ilp::ilp_problem_t *prob,
    GRBModel *model, const hash_map<ilp::variable_idx_t, GRBVar> &vars,
    const std::string &name) const
{
    std::vector<double> values(prob->variables().size(), 0);
    GRBVar *p_vars = model->getVars();
    double *p_values = model->get(GRB_DoubleAttr_X, p_vars, values.size());

    for (int i = 0; i < prob->variables().size(); ++i)
        values[i] = p_values[i];

    delete p_vars;
    delete p_values;

    return ilp::ilp_solution_t(prob, ilp::SOLUTION_OPTIMAL, values);
}

#endif


ilp_solver_t* gurobi_t::generator_t::operator()(phillip_main_t *ph) const
{
    return new sol::gurobi_t(
        ph,
        ph->param_int("gurobi_thread_num"),
        ph->flag("activate_gurobi_log"));
}


}

}
