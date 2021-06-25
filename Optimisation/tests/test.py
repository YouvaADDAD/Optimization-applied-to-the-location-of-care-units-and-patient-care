import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import unittest
from src.fonctions import *


class TestPGCD(unittest.TestCase):

    def test_normal(self):
        self.assertEqual(my_gcd(987, 345), 3)
        self.assertEqual(my_gcd(345, 987), 3)

    def test_etendu(self):
        gcd, u, v = my_gcd_etendu(987, 345)
        self.assertEqual(u*987 + v*345, gcd)
        gcd, u, v = my_gcd_etendu(345, 987)
        self.assertEqual(u*987 + v*345, gcd)

if __name__ == '__main__':
    unittest.main()