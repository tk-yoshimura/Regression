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

        private readonly Matrix xt;

        public Regressor(Vector[] xs, Vector ys, bool intercept = true) :
            this(intercept
                ? Matrix.HConcat([Vector.Fill(xs.First().Dim, 1d), .. xs])
                : Matrix.HConcat([.. xs]),
                ys, intercept: false) { }

        public Regressor(Matrix xs, Vector ys, bool intercept = true) {
            if (xs.Rows != ys.Dim) {
                throw new ArgumentException("mismatch size", $"{nameof(xs)}:(N, features),{nameof(ys)}:(N)");
            }

            if (!intercept) {
                this.X = xs.Copy();
            }
            else {
                Matrix m = Matrix.Zero(xs.Rows, xs.Columns + 1);
                m[.., 1..] = xs;
                m[.., 0] = Vector.Fill(xs.Rows, 1d);
                this.X = m;
            }

            this.Y = ys.Copy();

            this.N = X.Rows;
            this.Features = X.Columns;

            this.xt = X.Transpose;
        }

        public virtual Vector ExecuteFitting(Vector? weights = null) {
            if (weights is null) {
                Vector parameters = Matrix.Solve(xt * X, xt * Y);

                return parameters;
            }
            else {
                if (weights.Dim != N) {
                    throw new ArgumentException("mismatch size.", nameof(weights));
                }

                Matrix wxt = xt.Copy();

                for (int j = 0; j < Features; j++) {
                    for (int i = 0; i < N; i++) {
                        wxt[j, i] *= weights[i];
                    }
                }

                Vector parameters = Matrix.Solve(wxt * X, wxt * Y);

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

            Vector errors = FittingValue(X, parameters) - Y;

            return errors;
        }

        public static Vector FittingValue(Matrix xs, Vector parameters) {
            return xs * parameters;
        }

        public override string ToString() {
            return $"{typeof(Regressor).Name} ({N}x{Features})";
        }
    }
}
