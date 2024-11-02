using Algebra;
using DoubleDouble;
using System.Diagnostics;

namespace Regression {
    [DebuggerDisplay("{ToString(),nq}")]
    public class Regressor {
        public Matrix X { get; private set; }

        public Vector Y { get; private set; }

        public int N { get; private set; }

        public int Features { get; private set; }

        private readonly Matrix x, xt;
        private readonly Vector y;
        private readonly ddouble? intercept;

        public Regressor(Vector[] xs, Vector ys, ddouble? intercept = null) :
            this(Matrix.HConcat([.. xs]), ys, intercept) { }

        public Regressor(Matrix xs, Vector ys, ddouble? intercept = null) {
            if (xs.Rows != ys.Dim) {
                throw new ArgumentException("mismatch size", $"{nameof(xs)}:(N, features),{nameof(ys)}:(N)");
            }

            this.X = xs.Copy();
            this.Y = ys.Copy();

            this.N = X.Rows;
            this.Features = X.Columns + 1;

            this.x = (intercept is null) ? Matrix.Concat(new Matrix[,] { { Matrix.Fill(N, 1, 1d), X } }) : X;
            this.xt = x.T;
            this.y = (intercept is null) ? ys.Copy() : ys - intercept.Value;
            this.intercept = intercept;
        }

        public virtual Vector Fit(Vector? weights = null) {
            if (weights is null) {
                Vector parameters = Matrix.SolvePositiveSymmetric(xt * x, xt * y, enable_check_symmetric: false);

                if (intercept is not null) {
                    parameters = Vector.Concat(intercept.Value, parameters);
                }

                return parameters;
            }
            else {
                if (weights.Dim != N) {
                    throw new ArgumentException("mismatch size.", nameof(weights));
                }

                Matrix wxt = xt.Copy();

                for (int j = 0; j < wxt.Rows; j++) {
                    for (int i = 0; i < N; i++) {
                        wxt[j, i] *= weights[i];
                    }
                }

                Vector parameters = Matrix.SolvePositiveSymmetric(wxt * x, wxt * y, enable_check_symmetric: false);

                if (intercept is not null) {
                    parameters = Vector.Concat(intercept.Value, parameters);
                }

                return parameters;
            }
        }

        public ddouble Cost(Vector parameters) {
            if (parameters.Dim != Features) {
                throw new ArgumentException("invalid size", nameof(parameters));
            }

            Vector errors = Error(parameters);
            ddouble cost = errors.SquareNorm;

            return cost;
        }

        public Vector Error(Vector parameters) {
            if (parameters.Dim != Features) {
                throw new ArgumentException("invalid size", nameof(parameters));
            }

            Vector errors = Regress(X, parameters) - Y;

            return errors;
        }

        public static Vector Regress(Matrix xs, Vector parameters) {
            return xs * parameters[1..] + parameters[0];
        }

        public override string ToString() {
            return $"{typeof(Regressor).Name} ({N}x{Features})";
        }
    }
}
