using Algebra;
using DoubleDouble;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Regression;

namespace RegressionTests {
    [TestClass()]
    public class RegressorTests {
        [TestMethod()]
        public void WithInterceptTest() {
            static ddouble f(ddouble x, ddouble y) {
                return 0.5 + 0.25 * x + 0.125 * y + 0.0625 * x * y + 2 * x * x + 4 * y * y;
            }

            (Vector x, Vector y) = Vector.MeshGrid(new ddouble[] { 0, 1, 2, 3, 4 }, new ddouble[] { 1, 2, 3, 4 });
            Vector z = (f, (x, y));

            Regressor fitter = new([x, y, x * y, x * x, y * y], z);

            Assert.AreEqual(20, fitter.N);
            Assert.AreEqual(6, fitter.Features);

            Vector param = fitter.Fit();

            Vector expected = new ddouble[] { 0.5, 0.25, 0.125, 0.0625, 2, 4 };

            Assert.IsTrue((param - expected).Norm < 1e-25);

            Assert.IsTrue(fitter.Error(param).Norm < 1e-25);

            Assert.IsTrue((Regressor.Regress(fitter.X, param) - fitter.Y).Norm < 1e-25);
        }

        [TestMethod()]
        public void WithoutInterceptTest() {
            static ddouble f(ddouble x, ddouble y) {
                return 0.25 * x + 0.125 * y + 0.0625 * x * y + 2 * x * x + 4 * y * y;
            }

            (Vector x, Vector y) = Vector.MeshGrid(new ddouble[] { 0, 1, 2, 3, 4 }, new ddouble[] { 1, 2, 3, 4 });
            Vector z = (f, (x, y));

            Regressor fitter = new([x, y, x * y, x * x, y * y], z, intercept: 0);

            Assert.AreEqual(20, fitter.N);
            Assert.AreEqual(6, fitter.Features);

            Vector param = fitter.Fit();

            Vector expected = new ddouble[] { 0, 0.25, 0.125, 0.0625, 2, 4 };

            Assert.IsTrue((param - expected).Norm < 1e-25);

            Assert.AreEqual(0d, param[0]);

            Assert.IsTrue(fitter.Error(param).Norm < 1e-25);

            Assert.IsTrue((Regressor.Regress(fitter.X, param) - fitter.Y).Norm < 1e-25);
        }

        [TestMethod()]
        public void WithInterceptNonZeroTest() {
            static ddouble f(ddouble x, ddouble y) {
                return -2.5 + 0.25 * x + 0.125 * y + 0.0625 * x * y + 2 * x * x + 4 * y * y;
            }

            (Vector x, Vector y) = Vector.MeshGrid(new ddouble[] { 0, 1, 2, 3, 4 }, new ddouble[] { 1, 2, 3, 4 });
            Vector z = (f, (x, y));

            Regressor fitter = new([x, y, x * y, x * x, y * y], z, intercept: -2.5);

            Assert.AreEqual(20, fitter.N);
            Assert.AreEqual(6, fitter.Features);

            Vector param = fitter.Fit();

            Vector expected = new ddouble[] { -2.5, 0.25, 0.125, 0.0625, 2, 4 };

            Assert.IsTrue((param - expected).Norm < 1e-25);

            Assert.AreEqual(-2.5, param[0]);

            Assert.IsTrue(fitter.Error(param).Norm < 1e-25);

            Assert.IsTrue((Regressor.Regress(fitter.X, param) - fitter.Y).Norm < 1e-25);
        }

        [TestMethod()]
        public void WithInterceptWeightedTest() {
            static ddouble f(ddouble x, ddouble y) {
                return 0.5 + 0.25 * x + 0.125 * y + 0.0625 * x * y + 2 * x * x + 4 * y * y;
            }

            (Vector x, Vector y) = Vector.MeshGrid(new ddouble[] { 0, 1, 2, 3, 4 }, new ddouble[] { 1, 2, 3, 4 });
            Vector z = (f, (x, y));

            Vector w = Vector.Fill(x.Dim, 1d);

            z[12] = 800;
            w[12] = 0;

            Regressor fitter = new([x, y, x * y, x * x, y * y], z);

            Assert.AreEqual(20, fitter.N);
            Assert.AreEqual(6, fitter.Features);

            Vector param = fitter.Fit(w);

            Vector expected = new ddouble[] { 0.5, 0.25, 0.125, 0.0625, 2, 4 };

            Assert.IsTrue((param - expected).Norm < 1e-25);
        }

        [TestMethod()]
        public void WithoutInterceptWeightedTest() {
            static ddouble f(ddouble x, ddouble y) {
                return 0.25 * x + 0.125 * y + 0.0625 * x * y + 2 * x * x + 4 * y * y;
            }

            (Vector x, Vector y) = Vector.MeshGrid(new ddouble[] { 0, 1, 2, 3, 4 }, new ddouble[] { 1, 2, 3, 4 });
            Vector z = (f, (x, y));

            Vector w = Vector.Fill(x.Dim, 1d);

            z[12] = 800;
            w[12] = 0;

            Regressor fitter = new([x, y, x * y, x * x, y * y], z, intercept: 0);

            Assert.AreEqual(20, fitter.N);
            Assert.AreEqual(6, fitter.Features);

            Vector param = fitter.Fit(w);

            Vector expected = new ddouble[] { 0, 0.25, 0.125, 0.0625, 2, 4 };

            Assert.IsTrue((param - expected).Norm < 1e-25);

            Assert.AreEqual(0d, param[0]);
        }

        [TestMethod()]
        public void WithInterceptNonZeroWeightedTest() {
            static ddouble f(ddouble x, ddouble y) {
                return -2.5 + 0.25 * x + 0.125 * y + 0.0625 * x * y + 2 * x * x + 4 * y * y;
            }

            (Vector x, Vector y) = Vector.MeshGrid(new ddouble[] { 0, 1, 2, 3, 4 }, new ddouble[] { 1, 2, 3, 4 });
            Vector z = (f, (x, y));

            Vector w = Vector.Fill(x.Dim, 1d);

            z[12] = 800;
            w[12] = 0;

            Regressor fitter = new([x, y, x * y, x * x, y * y], z, intercept: -2.5);

            Assert.AreEqual(20, fitter.N);
            Assert.AreEqual(6, fitter.Features);

            Vector param = fitter.Fit(w);

            Vector expected = new ddouble[] { -2.5, 0.25, 0.125, 0.0625, 2, 4 };

            Assert.IsTrue((param - expected).Norm < 1e-25);

            Assert.AreEqual(-2.5, param[0]);
        }
    }
}