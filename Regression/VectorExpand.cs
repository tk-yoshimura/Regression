using Algebra;
using DoubleDouble;

namespace Regression {
    public static class VectorExpand {
        public static Vector Square(this Vector v) {
            return Vector.Func(v, ddouble.Square);
        }

        public static Vector Cube(this Vector v) {
            return Vector.Func(v, ddouble.Cube);
        }

        public static Vector Pow(this Vector v, int n) {
            return Vector.Func(v, x => ddouble.Pow(x, n));
        }
    }
}
