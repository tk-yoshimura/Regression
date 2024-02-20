using Algebra;
using DoubleDouble;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Regression;

namespace RegressionTests {
    [TestClass()]
    public class RobustRegressorTests {
        [TestMethod()]
        public void WithInterceptWeightedTest() {
            static ddouble f(ddouble x, ddouble y) {
                return 0.5 + 0.25 * x + 0.125 * y + 0.0625 * x * y + 2 * x * x + 4 * y * y;
            }

            Vector x = new ddouble[] { 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 };
            Vector y = new ddouble[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4 };
            Vector w = Vector.Fill(x.Dim, 1d);

            Vector z = Vector.Func(x, y, f);

            z[12] = 800;
            w[12] = 0;

            RobustRegressor fitter = new([x, y, x * y, x * x, y * y], z);

            Assert.AreEqual(20, fitter.N);
            Assert.AreEqual(6, fitter.Features);

            Assert.IsTrue(fitter.W[11] == 1d);
            Assert.IsFalse(fitter.W[12] == 0d);

            Vector param = fitter.ExecuteFitting();

            Vector expected = new ddouble[] { 0.5, 0.25, 0.125, 0.0625, 2, 4 };

            Assert.IsTrue((param - expected).Norm < 1e-25);

            Assert.IsTrue(fitter.W[11] > 0d);
            Assert.IsTrue(fitter.W[12] == 0d);
        }

        [TestMethod()]
        public void WithoutInterceptWeightedTest() {
            static ddouble f(ddouble x, ddouble y) {
                return 0.25 * x + 0.125 * y + 0.0625 * x * y + 2 * x * x + 4 * y * y;
            }

            Vector x = new ddouble[] { 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 };
            Vector y = new ddouble[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4 };
            Vector w = Vector.Fill(x.Dim, 1d);

            Vector z = Vector.Func(x, y, f);

            z[12] = 800;
            w[12] = 0;

            RobustRegressor fitter = new([x, y, x * y, x * x, y * y], z, intercept: false);

            Assert.AreEqual(20, fitter.N);
            Assert.AreEqual(5, fitter.Features);

            Assert.IsTrue(fitter.W[11] == 1d);
            Assert.IsFalse(fitter.W[12] == 0d);

            Vector param = fitter.ExecuteFitting();

            Vector expected = new ddouble[] { 0.25, 0.125, 0.0625, 2, 4 };

            Assert.IsTrue((param - expected).Norm < 1e-25);

            Assert.IsTrue(fitter.W[11] > 0d);
            Assert.IsTrue(fitter.W[12] == 0d);
        }
    }
}