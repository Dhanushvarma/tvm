/* Copyright 2022 CNRS-AIST JRL and CNRS-UM LIRMM */
// TVM_USE_LEXLS comes from the generated supported_solvers.h; TVM_WITH_LEXLS, the CMake option, is no macro, so testing
// it left this file without a test case
#include <tvm/supported_solvers.h>

#ifdef TVM_USE_LEXLS
#  include <tvm/solver/LexLSHierarchicalLeastSquareSolver.h>
#endif

#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/constraint/BasicLinearConstraint.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/scheme/HierarchicalLeastSquares.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/task_dynamics/None.h>

#include <Eigen/SVD>

#include <cmath>
#include <utility>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/EigenDoctest.h"

using namespace tvm;
using namespace tvm::constraint;
using namespace tvm::requirements;
using namespace tvm::solver;
using namespace Eigen;

MatrixXd pinv(const MatrixConstRef & M, double eps = 1e-10)
{
  auto svd = M.jacobiSvd(ComputeThinU | ComputeThinV);
  svd.setThreshold(eps);
  int r = svd.rank();
  return svd.matrixV().leftCols(r) * svd.singularValues().head(r).cwiseInverse().asDiagonal()
         * svd.matrixU().leftCols(r).transpose();
}

/** The lexicographic solution of levels (A_k, b_k) by projections, each level solved in the null space of the ones above
 * (the hand computation of the test cases below, for any number of levels) */
VectorXd lexicographic(const std::vector<std::pair<MatrixXd, VectorXd>> & levels, Index n)
{
  VectorXd x = VectorXd::Zero(n);
  MatrixXd P = MatrixXd::Identity(n, n);
  MatrixXd stacked(0, n);
  for(const auto & [A, b] : levels)
  {
    x += pinv(A * P) * (b - A * x);
    stacked.conservativeResize(stacked.rows() + A.rows(), n);
    stacked.bottomRows(A.rows()) = A;
    P = MatrixXd::Identity(n, n) - pinv(stacked) * stacked;
  }
  return x;
}

#ifdef TVM_USE_LEXLS
TEST_CASE("LexLSHierarchicalLeastSquareSolver")
{
  VariablePtr x = Space(6).createVariable("x");
  MatrixXd A0(3, 6), A1(3, 6), A2(3, 6);
  VectorXd b0(3), b1(3), b2(3);
  // rank(A0)=2
  A0 = MatrixXd::Random(3, 2) * MatrixXd::Random(2, 6);
  // rank(A1) = 3 but rank(A1 projected on the nullspace of A0) = 2
  A1 << MatrixXd::Random(1, 3) * A0, MatrixXd::Random(2, 6);
  A1 = MatrixXd::Random(3, 3) * A1;
  A2.setRandom();
  b0.setRandom();
  b1.setRandom();
  b2.setRandom();
  auto c0 = std::make_shared<BasicLinearConstraint>(A0, x, b0, constraint::Type::EQUAL);
  auto c1a = std::make_shared<BasicLinearConstraint>(A1.row(0), x, b1.head(1), constraint::Type::EQUAL);
  auto c1b = std::make_shared<BasicLinearConstraint>(A1.row(1), x, b1.segment(1, 1), constraint::Type::EQUAL);
  auto c1c = std::make_shared<BasicLinearConstraint>(A1.row(2), x, b1.tail(1), constraint::Type::EQUAL);
  auto c2 = std::make_shared<BasicLinearConstraint>(A2, x, b2, constraint::Type::EQUAL);

  auto r0 = std::make_shared<SolvingRequirementsWithCallbacks>(PriorityLevel(0));
  auto r1 = std::make_shared<SolvingRequirementsWithCallbacks>(PriorityLevel(1));
  auto r2 = std::make_shared<SolvingRequirementsWithCallbacks>(PriorityLevel(2));

  LexLSHierarchicalLeastSquareSolver solver(LexLSHLSSolverOptions{});
  VariableVector vars(x);
  solver.startBuild(vars, {3, 3, 3}, {0, 0, 0}, false);
  solver.addConstraint(c0, r0);
  solver.addConstraint(c1a, r1);
  solver.addConstraint(c1b, r1);
  solver.addConstraint(c1c, r1);
  solver.addConstraint(c2, r2);
  solver.finalizeBuild();

  solver.solve();

  // Compute the solution by hand
  MatrixXd P0 = MatrixXd::Identity(6, 6) - pinv(A0) * A0;
  MatrixXd A01(6, 6);
  A01 << A0, A1;
  MatrixXd P1 = MatrixXd::Identity(6, 6) - pinv(A01) * A01;
  VectorXd dx0 = pinv(A0) * b0;
  VectorXd dx1 = pinv(A1 * P0) * (b1 - A1 * dx0);
  VectorXd dx2 = pinv(A2 * P1) * (b2 - A2 * (dx0 + dx1));
  VectorXd x0 = dx0 + dx1 + dx2;

  FAST_CHECK_EQ(solver.result(), Approx(x0).epsilon(1e-10));
}

TEST_CASE("LexLSHierarchicalLeastSquareSolver min norm")
{
  VariablePtr x = Space(6).createVariable("x");
  MatrixXd A0(3, 6), A1(3, 6);
  VectorXd b0(3), b1(3);
  // rank(A0)=2
  A0 = MatrixXd::Random(3, 2) * MatrixXd::Random(2, 6);
  // rank(A1) = 3 but rank(A1 projected on the nullspace of A0) = 2
  A1 << MatrixXd::Random(1, 3) * A0, MatrixXd::Random(2, 6);
  A1 = MatrixXd::Random(3, 3) * A1;
  b0.setRandom();
  b1.setRandom();
  auto c0 = std::make_shared<BasicLinearConstraint>(A0, x, b0, constraint::Type::EQUAL);
  auto c1 = std::make_shared<BasicLinearConstraint>(A1, x, b1, constraint::Type::EQUAL);

  auto r0 = std::make_shared<SolvingRequirementsWithCallbacks>(PriorityLevel(0));
  auto r1 = std::make_shared<SolvingRequirementsWithCallbacks>(PriorityLevel(1));

  LexLSHierarchicalLeastSquareSolver solver(LexLSHLSSolverOptions{});
  VariableVector vars(x);
  solver.startBuild(vars, {3, 3, 6}, {0, 0, 0}, false);
  solver.addConstraint(c0, r0);
  solver.addConstraint(c1, r1);
  solver.setMinimumNorm(); // Equivalent to A2 = I and b2 = 0
  solver.finalizeBuild();

  solver.solve();

  // Compute the solution by hand
  MatrixXd P0 = MatrixXd::Identity(6, 6) - pinv(A0) * A0;
  MatrixXd A01(6, 6);
  A01 << A0, A1;
  MatrixXd P1 = MatrixXd::Identity(6, 6) - pinv(A01) * A01;
  VectorXd dx0 = pinv(A0) * b0;
  VectorXd dx1 = pinv(A1 * P0) * (b1 - A1 * dx0);
  VectorXd dx2 = pinv(P1) * (-(dx0 + dx1));
  VectorXd x0 = dx0 + dx1 + dx2;

  FAST_CHECK_EQ(solver.result(), Approx(x0).epsilon(1e-10));
}

TEST_CASE("HierarchicalLeastSquares")
{
  SUBCASE("Equality only")
  {
    VariablePtr x = Space(6).createVariable("x");
    MatrixXd A0(3, 6), A1(3, 6), A2(3, 6);
    VectorXd b0(3), b1(3), b2(3);
    // rank(A0)=2
    A0 = MatrixXd::Random(3, 2) * MatrixXd::Random(2, 6);
    // rank(A1) = 3 but rank(A1 projected on the nullspace of A0) = 2
    A1 << MatrixXd::Random(1, 3) * A0, MatrixXd::Random(2, 6);
    A1 = MatrixXd::Random(3, 3) * A1;
    A2.setRandom();
    b0.setRandom();
    b1.setRandom();
    b2.setRandom();

    LinearizedControlProblem pb;
    pb.add(A0 * x - b0 == 0., {PriorityLevel(0)});
    pb.add(A1 * x - b1 == 0., {PriorityLevel(1)});
    pb.add(A2 * x - b2 == 0., {PriorityLevel(2)});

    scheme::HierarchicalLeastSquares solver(LexLSHLSSolverOptions{});

    solver.solve(pb);

    // Compute the solution by hand
    MatrixXd P0 = MatrixXd::Identity(6, 6) - pinv(A0) * A0;
    MatrixXd A01(6, 6);
    A01 << A0, A1;
    MatrixXd P1 = MatrixXd::Identity(6, 6) - pinv(A01) * A01;
    VectorXd dx0 = pinv(A0) * b0;
    VectorXd dx1 = pinv(A1 * P0) * (b1 - A1 * dx0);
    VectorXd dx2 = pinv(A2 * P1) * (b2 - A2 * (dx0 + dx1));
    VectorXd x0 = dx0 + dx1 + dx2;

    FAST_CHECK_EQ(x->value(), Approx(x0).epsilon(1e-10));
  }

  SUBCASE("Inequality only")
  {
    VariablePtr x = Space(1).createVariable("x");
    VariablePtr y = Space(1).createVariable("y");

    LinearizedControlProblem pb;
    pb.add(-1. <= x <= 1., {PriorityLevel(0)});
    pb.add(2. <= x <= 3., {PriorityLevel(1)});
    pb.add(-1. <= y <= 1., {PriorityLevel(2)});
    pb.add(x + y <= 0., {PriorityLevel(4)});

    scheme::HierarchicalLeastSquares solver(LexLSHLSSolverOptions().feasibleFirstLevel(true));

    solver.solve(pb);
    FAST_CHECK_EQ(x->value()[0], doctest::Approx(1).epsilon(1e-10));
    FAST_CHECK_EQ(y->value()[0], doctest::Approx(-1).epsilon(1e-10));
  }

  SUBCASE("Runtime changes")
  {
    // A weight changed, a task removed and a task added between solves: the scheme rebuilds its solver data each time
    const Index n = 8;
    VariablePtr x = Space(static_cast<int>(n)).createVariable("x");
    MatrixXd A0(3, n), A1(3, n), A2a(3, n), A2b(3, n), A3(2, n);
    VectorXd b0(3), b1(3), b2a(3), b2b(3), b3(2);
    // rank(A0) = 2; A1 projected on the null space of A0 has rank 2, which leaves 4 degrees of freedom to level 2,
    // where tasks a and b (6 rows) compete through their weights
    A0 = MatrixXd::Random(3, 2) * MatrixXd::Random(2, n);
    A1 << MatrixXd::Random(1, 3) * A0, MatrixXd::Random(2, n);
    A2a.setRandom();
    A2b.setRandom();
    A3.setRandom();
    b0.setRandom();
    b1.setRandom();
    b2a.setRandom();
    b2b.setRandom();
    b3.setRandom();

    LinearizedControlProblem pb;
    pb.add(A0 * x - b0 == 0., {PriorityLevel(0)});
    pb.add(A1 * x - b1 == 0., {PriorityLevel(1)});
    auto ta = pb.add(A2a * x - b2a == 0., {PriorityLevel(2), Weight(1.)});
    auto tb = pb.add(A2b * x - b2b == 0., {PriorityLevel(2), Weight(4.)});

    scheme::HierarchicalLeastSquares solver(LexLSHLSSolverOptions{});

    // Level 2 minimises |A2a x - b2a|^2 + w |A2b x - b2b|^2: its rows are scaled by the square roots of the weights
    auto level2 = [&](double wa, double wb)
    {
      MatrixXd A(6, n);
      VectorXd b(6);
      A << std::sqrt(wa) * A2a, std::sqrt(wb) * A2b;
      b << std::sqrt(wa) * b2a, std::sqrt(wb) * b2b;
      return std::make_pair(A, b);
    };
    // Each level's fitted value A_k x is unique at the lexicographic optimum, while x need not be: the solver may leave
    // a free direction elsewhere than the minimum-norm solution of the hand computation
    auto check = [&](const std::vector<std::pair<MatrixXd, VectorXd>> & levels)
    {
      const VectorXd xs = lexicographic(levels, n);
      for(const auto & [A, b] : levels)
      {
        FAST_CHECK_EQ(A * x->value(), Approx(A * xs).epsilon(1e-8));
      }
    };
    solver.solve(pb);
    check({{A0, b0}, {A1, b1}, level2(1., 4.)});

    tb->requirements.weight() = 9.;
    solver.solve(pb);
    check({{A0, b0}, {A1, b1}, level2(1., 9.)});

    // Without task a, task b (3 rows) leaves one degree of freedom to level 3
    pb.remove(*ta);
    solver.solve(pb);
    check({{A0, b0}, {A1, b1}, {A2b, b2b}});

    pb.add(A3 * x - b3 == 0., {PriorityLevel(3)});
    solver.solve(pb);
    check({{A0, b0}, {A1, b1}, {A2b, b2b}, {A3, b3}});
    // With the four levels, x is unique: the full solution too
    FAST_CHECK_EQ(x->value(), Approx(lexicographic({{A0, b0}, {A1, b1}, {A2b, b2b}, {A3, b3}}, n)).epsilon(1e-8));
  }

  SUBCASE("Crossed bounds")
  {
    // Two bounds on x at level 0 that cross, as a joint limit damper asking more speed than the velocity limit allows:
    // they meet at their midpoint, where LexLS would reject the row. With feasibleFirstLevel, the same as simple bounds
    for(bool feasibleFirstLevel : {false, true})
    {
      VariablePtr x = Space(1).createVariable("x");
      VariablePtr y = Space(1).createVariable("y");

      LinearizedControlProblem pb;
      pb.add(2. <= x <= 5., {PriorityLevel(0)});
      pb.add(-5. <= x <= 1., {PriorityLevel(0)});
      pb.add(x + y == 0., {PriorityLevel(1)});

      scheme::HierarchicalLeastSquares solver(LexLSHLSSolverOptions().feasibleFirstLevel(feasibleFirstLevel));
      CHECK_NOTHROW(solver.solve(pb));
      FAST_CHECK_EQ(x->value()[0], doctest::Approx(1.5).epsilon(1e-10));
      FAST_CHECK_EQ(y->value()[0], doctest::Approx(-1.5).epsilon(1e-10));
    }
  }

  SUBCASE("Regularization")
  {
    // An ill-conditioned level, diag(1, 1e-3) x = (1, 1): undamped x = (1, 1000). Damped by mu, each coordinate is
    // a b / (a^2 + mu^2), the minimiser of |A x - b|^2 + mu^2 |x|^2
    const double mu = 0.01;
    for(bool damped : {false, true})
    {
      VariablePtr x = Space(2).createVariable("x");
      MatrixXd A = Vector2d(1., 1e-3).asDiagonal();
      VectorXd b = Vector2d(1., 1.);
      LinearizedControlProblem pb;
      pb.add(A * x - b == 0., {PriorityLevel(1)});
      LexLSHLSSolverOptions options;
      if(damped)
      {
        options.regularization_type(LexLS::REGULARIZATION_TIKHONOV);
        options.regularizationFactors({0., mu});
      }
      scheme::HierarchicalLeastSquares solver(options);
      solver.solve(pb);
      const Vector2d expected = damped ? Vector2d(1. / (1. + mu * mu), 1e-3 / (1e-6 + mu * mu)) : Vector2d(1., 1000.);
      FAST_CHECK_EQ(x->value(), Approx(expected).epsilon(1e-8));
    }
  }

  SUBCASE("Empty levels")
  {
    // Levels need not start at 0 nor follow each other: mc_rtc puts constraints at 0 and tasks from 1, and a level
    // empties when its last task is removed. Two tasks pull x0 to opposite targets at levels 1 and 2, a third one holds
    // the other coordinates at level 3 (posture tasks selecting joints by zero anisotropic weights)
    for(bool autoDamping : {false, true})
    {
      const Index n = 6;
      VariablePtr x = Space(static_cast<int>(n)).createVariable("x");
      const MatrixXd I = MatrixXd::Identity(n, n);
      VectorXd first = VectorXd::Zero(n), others = VectorXd::Ones(n);
      first(0) = 1.;
      others(0) = 0.;
      const VectorXd up = VectorXd::Constant(n, 0.3), down = VectorXd::Constant(n, -0.3), rest = VectorXd::Zero(n);

      LinearizedControlProblem pb;
      auto tUp = pb.add(I * x - up == 0., {PriorityLevel(1), AnisotropicWeight(first)});
      pb.add(I * x - down == 0., {PriorityLevel(2), AnisotropicWeight(first)});
      pb.add(I * x - rest == 0., {PriorityLevel(3), Weight(1e-3), AnisotropicWeight(others)});

      scheme::HierarchicalLeastSquares solver(LexLSHLSSolverOptions{},
                                              scheme::HierarchicalLeastSquaresOptions().autoDamping(autoDamping));
      VectorXd expected = VectorXd::Zero(n);
      expected(0) = 0.3;
      solver.solve(pb);
      FAST_CHECK_EQ(x->value(), Approx(expected).epsilon(1e-8));

      // Without the level-1 task, levels 0 and 1 are empty
      pb.remove(*tUp);
      solver.solve(pb);
      expected(0) = -0.3;
      FAST_CHECK_EQ(x->value(), Approx(expected).epsilon(1e-8));
    }
  }
}
#endif
