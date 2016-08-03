/* -*- coding: utf-8 -*- */

#include <ctime>
#include <thread>
#include <algorithm>

#include "./phillip.h"


namespace phil
{


int phillip_main_t::ms_verboseness = VERBOSE_1;
const std::string phillip_main_t::VERSION = "phil.3.17";


phillip_main_t::phillip_main_t()
: m_lhs_enumerator(NULL), m_ilp_convertor(NULL), m_ilp_solver(NULL),
  m_input(NULL), m_lhs(NULL), m_ilp(NULL),
  m_time_for_enumerate(0), m_time_for_convert(0), m_time_for_convert_gold(0),
  m_time_for_solve(0), m_time_for_solve_gold(0),
  m_time_for_learn(0), m_time_for_infer(0)
{}


phillip_main_t::~phillip_main_t()
{
    if (m_lhs_enumerator != NULL) delete m_lhs_enumerator;
    if (m_ilp_convertor != NULL)  delete m_ilp_convertor;
    if (m_ilp_solver != NULL)     delete m_ilp_solver;

    if (m_input != NULL) delete m_input;
    if (m_lhs != NULL)   delete m_lhs;
    if (m_ilp != NULL)   delete m_ilp;
}


phillip_main_t* phillip_main_t::duplicate() const
{
    phillip_main_t *out = new phillip_main_t();
    out->set_lhs_enumerator(m_lhs_enumerator->duplicate(out));
    out->set_ilp_convertor(m_ilp_convertor->duplicate(out));
    out->set_ilp_solver(m_ilp_solver->duplicate(out));

    out->m_params.insert(m_params.begin(), m_params.end());
    out->m_flags.insert(m_flags.begin(), m_flags.end());
    out->m_timeout_lhs = m_timeout_lhs;
    out->m_timeout_ilp = m_timeout_ilp;
    out->m_timeout_sol = m_timeout_sol;

    return out;
}


std::ofstream* _open_file(const std::string &path, std::ios::openmode mode)
{
    if (not path.empty())
    {
        util::mkdir(util::get_directory_name(path));

        std::ofstream *fo = new std::ofstream(path.c_str(), mode);
        if (fo->good())
            return fo;
        else
        {
            util::print_error_fmt("Cannot open file: \"%s\"", path.c_str());
            delete fo;
        }
    }
    return NULL;
}


int phillip_main_t::infer(const lf::input_t &input)
{
    reset_for_inference();
    set_input(input);

    auto begin = std::chrono::system_clock::now();

    execute_enumerator();
    execute_convertor();
    execute_solver();

    m_time_for_infer = util::duration_time(begin);

    std::ofstream *fo(NULL);
    if ((fo = _open_file(param("path_out"), std::ios::out | std::ios::app)) != NULL)
    {
        for (auto sol = m_sol.begin(); sol != m_sol.end(); ++sol)
            sol->print_graph(fo);
        delete fo;
    }

    return 0;
}


int phillip_main_t::learn(const lf::input_t &input)
{
    auto get_path_for_gold = [this](const std::string &key) -> std::string
    {
        std::string path = param(key);
        if (not path.empty())
        {
            int idx = path.rfind('.');
            if (idx > 0)
                path = path.substr(0, idx) + ".gold" + path.substr(idx);
            else
                path += ".gold";
        }
        return path;
    };

    reset_for_inference();
    set_input(input);

    auto begin = std::chrono::system_clock::now();

    execute_enumerator();

    // Purely predict (with tie-care).
    assert(flag("kbest"));

    if(param_float("margin", 0.0) > 0.0)
        set_param("kbest_k", "2");
    else
        set_param("kbest_k", "1");

    erase_flag("get_pseudo_positive");
    execute_convertor();
    execute_solver();

    util::xml_element_t elem("learn", "");
    int num_updates = 0;

    // Check if the required literals are included in a top-scored hypotheses.

    // Identify a good solution. If there is no good solution, then get it through inference.
    int good_sol = -1;
    std::vector<int> bad_sols;

    for(auto i=0; i<m_sol.size(); i++) {
        if(m_sol[i].contains(get_latent_hypotheses_set()->requirements())) {
            good_sol = i;
        } else {
            bad_sols.push_back(i);
        }
    }

    if(-1 == good_sol) {
        util::print_console("No good solution found. Run latent variable completion...");
        set_param("kbest_k", "1");
        set_flag("get_pseudo_positive");
        execute_convertor();
        execute_solver();

        good_sol = m_sol.size()-1;
    }

    // Update the weights.
    for(auto bad_sol: bad_sols) {
        if(m_sol[bad_sol].value_of_objective_function() - m_sol[good_sol].value_of_objective_function() > param_float("margin", 0.0)) continue;

        util::print_console_fmt("Margin: %f", m_sol[bad_sol].value_of_objective_function() - m_sol[good_sol].value_of_objective_function());

        m_ilp_convertor->tune(m_sol[bad_sol], m_sol[good_sol], &elem);
        num_updates++;
    }

    m_time_for_learn = util::duration_time(begin);

    std::ofstream *fo(NULL);
    if ((fo = _open_file(param("path_out"), std::ios::out | std::ios::app)) != NULL)
    {
        if (not flag("omit_proof_graph_from_xml"))
        {
          for (auto sol = m_sol.begin(); sol != m_sol.end(); ++sol)
              sol->print_graph(fo);
        }
        elem.print(fo);
        delete fo;
    }

    return std::min(num_updates, 1);
}


void phillip_main_t::execute_enumerator(
    pg::proof_graph_t **out_lhs, duration_time_t *out_time,
    const std::string &path_out_xml)
{
    IF_VERBOSE_2("Generating latent-hypotheses-set...");

    if ((*out_lhs) != NULL) delete m_lhs;

    auto begin = std::chrono::system_clock::now();
    (*out_lhs) = m_lhs_enumerator->execute();
    (*out_time) = util::duration_time(begin);

    IF_VERBOSE_2(
        m_lhs->has_timed_out() ?
        "Interrupted generating latent-hypotheses-set." :
        "Completed generating latent-hypotheses-set.");

    if (not path_out_xml.empty())
    {
        std::ios::openmode mode = std::ios::out | std::ios::app;
        std::ofstream *fo = _open_file(path_out_xml, mode);
        if (fo != NULL)
        {
            m_lhs->print(fo);
            delete fo;
        }
    }
}


void phillip_main_t::execute_convertor(
    ilp::ilp_problem_t **out_ilp, duration_time_t *out_time,
    const std::string &path_out_xml)
{
    IF_VERBOSE_2("Converting LHS into linear-programming-problems...");

    auto begin = std::chrono::system_clock::now();
    (*out_ilp) = m_ilp_convertor->execute();
    (*out_time) = util::duration_time(begin);

    IF_VERBOSE_2(
        m_ilp->has_timed_out() ?
        "Interrupted convertion into linear-programming-problems." :
        "Completed convertion into linear-programming-problems.");

    if (not path_out_xml.empty())
    {
        std::ios::openmode mode = std::ios::out | std::ios::app;
        std::ofstream *fo = _open_file(path_out_xml, mode);
        if (fo != NULL)
        {
            m_ilp->print(fo);
            delete fo;
        }
    }
}


void phillip_main_t::execute_solver(
    std::vector<ilp::ilp_solution_t> *out_sols,
    duration_time_t *out_time,
    const std::string &path_out_xml)
{
    IF_VERBOSE_2("Solving...");

    auto begin = std::chrono::system_clock::now();
    m_ilp_solver->execute(out_sols);
    (*out_time) = util::duration_time(begin);

    IF_VERBOSE_2("Completed inference.");

    if (not path_out_xml.empty())
    {
        std::ios::openmode mode = std::ios::out | std::ios::app;
        std::ofstream *fo = _open_file(path_out_xml, mode);
        if (fo != NULL)
        {
            for (auto sol = m_sol.begin(); sol != m_sol.end(); ++sol)
                sol->print(fo);
            delete fo;
        }
    }
}


void phillip_main_t::write_accuracy(int num_correct, int num_total) const {

    std::ofstream *fo(NULL);
    if ((fo = _open_file(param("path_out"), std::ios::out | std::ios::app)) != NULL) {
        (*fo) << "<performance>" << std::endl
              << "<correct>" << num_correct << "</correct>" << std::endl
              << "<total>" << num_total << "</total>" << std::endl
              << "<accuracy>" << (float)num_correct / num_total << "</accuracy>" << std::endl
              << "</performance>" << std::endl
              ;

        delete fo;
    }

}


void phillip_main_t::write_tuned_parameters() const {

    std::ofstream *fo(NULL);
    if ((fo = _open_file(param("tuned_param_out"), std::ios::out)) != NULL) {
        ilp_convertor()->print_tuned_parameters(fo);
        delete fo;
    }

}


void phillip_main_t::load_tuned_parameters() {

    if(param("tuned_param_in") != "") {
        util::print_console_fmt("Loading weight values from %s...", param("tuned_param_in").c_str());
        std::ifstream fi(param("tuned_param_in"));
        ilp_convertor()->load_tuned_parameters(&fi);
        fi.close();
    }

}


void phillip_main_t::write_header() const
{
    auto write = [this](std::ostream *os)
    {
        (*os) << "<phillip>" << std::endl;
        (*os) << "<configure>" << std::endl;
        (*os) << "<version>" << VERSION << "</version>" << std::endl;

        auto get_time_stamp_exe = []() -> std::string
        {
            int year, month, day, hour, min, sec;
            std::string out;
            util::beginning_time(&year, &month, &day, &hour, &min, &sec);
            switch (month)
            {
            case 1:  out = "Jan"; break;
            case 2:  out = "Feb"; break;
            case 3:  out = "Mar"; break;
            case 4:  out = "Apr"; break;
            case 5:  out = "May"; break;
            case 6:  out = "Jun"; break;
            case 7:  out = "Jul"; break;
            case 8:  out = "Aug"; break;
            case 9:  out = "Sep"; break;
            case 10: out = "Oct"; break;
            case 11: out = "Nov"; break;
            case 12: out = "Dec"; break;
            default: throw;
            }
            return out + util::format(" %2d %4d %02d:%02d:%02d", day, year, hour, min, sec);
        };

        (*os)
            << "<time_stamp compiled=\"" << util::format("%s %s", __DATE__, __TIME__)
            << "\" executed=\"" << get_time_stamp_exe()
            << "\"></time_stamp>" << std::endl;

        (*os)
            << "<components lhs=\"" << m_lhs_enumerator->repr()
            << "\" ilp=\"" << m_ilp_convertor->repr()
            << "\" sol=\"" << m_ilp_solver->repr()
            << "\"></components>" << std::endl;

        const kb::knowledge_base_t *base = kb::knowledge_base_t::instance();
        (*os)
            << "<knowledge_base path=\"" << base->filename()
            << "\" size=\"" << base->num_of_axioms()
            << "\" max_distance=\"" << base->get_max_distance()
            << "\"></knowledge_base>" << std::endl;

        (*os)
            << "<params timeout_lhs=\"" << timeout_lhs().get()
            << "\" timeout_ilp=\"" << timeout_ilp().get()
            << "\" timeout_sol=\"" << timeout_sol().get()
            << "\" timeout_all=\"" << timeout_all().get()
            << "\" verbose=\"" << verbose();

        for (auto it = m_params.begin(); it != m_params.end(); ++it)
            (*os) << "\" " << it->first << "=\"" << it->second;

        for (auto it = m_flags.begin(); it != m_flags.end(); ++it)
            (*os) << "\" " << (*it) << "=\"yes";

#ifdef DISABLE_CANCELING
        (*os) << "\" disable_canceling=\"yes";
#endif

#ifdef DISABLE_HARD_TERM
        (*os) << "\" disable_hard_term=\"yes";
#endif

        (*os) << "\"></params>" << std::endl;

        (*os) << "</configure>" << std::endl;
    };

    auto f_write = [&](const std::string &key)
    {
        std::ofstream *fo(NULL);
        if ((fo = _open_file(param(key), (std::ios::out | std::ios::trunc))) != NULL)
        {
            write(fo);
            delete fo;
        }
    };

    f_write("path_lhs_out");
    f_write("path_ilp_out");
    f_write("path_sol_out");
    f_write("path_out");
    write(&std::cout);
}


void phillip_main_t::write_footer() const
{
    auto write = [this](std::ostream *os)
    {
        (*os) << "</phillip>" << std::endl;
    };
    auto f_write = [&](const std::string &key)
    {
        std::ofstream *fo(NULL);
        if ((fo = _open_file(param(key), (std::ios::out | std::ios::app))) != NULL)
        {
            write(fo);
            delete fo;
        }
    };

    f_write("path_lhs_out");
    f_write("path_ilp_out");
    f_write("path_sol_out");
    f_write("path_out");
    write(&std::cout);
}


}
