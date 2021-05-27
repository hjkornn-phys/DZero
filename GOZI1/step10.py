import numpy as np
import unittest

if __name__ == "__main__":
    from step09 import Variable, square
    from step04 import numerical_diff

    class Squaretest(unittest.TestCase):
        def test_forward(self):
            x = Variable(np.array(2.0))
            y = square(x)
            expected = np.array(4.0)
            self.assertEqual(y.data, expected)

        def test_backward(self):
            x = Variable(np.array(3.0))
            y = square(x)
            y.backward()
            expected = np.array(6.0)
            self.assertEqual(x.grad, expected)

        def test_gradient_check(self):
            x = Variable(np.random.rand(1))
            y = square(x)
            y.backward()
            num_grad = numerical_diff(square, x)
            flg = np.allclose(x.grad, num_grad)
            self.assertTrue(flg)
    unittest.main()


else:
    from GOZI1.step09 import Variable, square
    from GOZI1.step04 import numerical_diff

    class Squaretest(unittest.TestCase):
        def test_forward(self):
            x = Variable(np.array(2.0))
            y = square(x)
            expected = np.array(4.0)
            self.assertEqual(y.data, expected)

        def test_backward(self):
            x = Variable(np.array(3.0))
            y = square(x)
            y.backward()
            expected = np.array(6.0)
            self.assertEqual(x.grad, expected)

        def test_gradient_check(self):
            x = Variable(np.random.rand(1))
            y = square(x)
            y.backward()
            num_grad = numerical_diff(square, x)
            flg = np.allclose(x.grad, num_grad)
            self.assertTrue(flg)
