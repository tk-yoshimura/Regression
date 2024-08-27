using Algebra;

namespace Regression {
    public class RobustRegressor : Regressor {
        public Vector W { get; private set; }

        public RobustRegressor(Vector[] xs, Vector ys, bool intercept = true)
            : base(xs, ys, intercept) {

            this.W = Vector.Fill(N, 1d);
        }

        public RobustRegressor(Matrix xs, Vector ys, bool intercept = true)
            : base(xs, ys, intercept) {

            this.W = Vector.Fill(N, 1d);
        }

        public sealed override Vector Fit(Vector? weights = null) {
            throw new InvalidOperationException();
        }

        public Vector Fit(int iter = 8, double eps = 1e-16) {
            if (!(eps > 0)) {
                throw new ArgumentOutOfRangeException(nameof(eps));
            }

            double err_threshold, inv_err;
            double[] weights = new double[N], errs = new double[N];

            Vector param = base.Fit();

            for (int i = 0; i < N; i++) {
                weights[i] = 1;
            }

            double s = 4;
            while (iter > 0) {
                Vector err = Error(param);
                for (int i = 0; i < N; i++) {
                    errs[i] = Math.Abs((double)err[i]);
                }

                double[] sort_err_list = (double[])errs.Clone();
                Array.Sort(sort_err_list);

                err_threshold = sort_err_list[N / 2] * s;
                if (err_threshold <= eps) {
                    break;
                }

                inv_err = 1 / (err_threshold + double.Epsilon);

                for (int i = 0; i < N; i++) {
                    double v = errs[i] * inv_err;
                    double r = double.Max(0d, 1 - v * v);
                    weights[i] = r * r;
                }

                param = base.Fit(new Vector(weights));

                iter--;
                s = double.Max(s * 0.75, 1.25);
            }

            W = weights;

            return param;
        }
    }
}
