using Algebra;
using DoubleDouble;

namespace Regression {
    public static class VectorExpand {
        public static Vector Square(this Vector v) {
            return (ddouble.Square, v);
        }

        public static Vector Cube(this Vector v) {
            return (ddouble.Cube, v);
        }

        public static Vector Pow(this Vector v, int n) {
            return (x => ddouble.Pow(x, n), v);
        }
    }
}
