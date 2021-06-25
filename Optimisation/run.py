
import os

from src.fonctions import *

import unittest
import tests.test

import numpy as np



def lancer_tests():
    return 

def q1():
    k = int(input("Entrez une valeur pour K: "))
    a = float(input("Entrer une valeur pour alpha: "))
    
    model = AffectationUniteSoin(k, a, random=None)
    model.solve()
    model.evaluateObjective()
    model.drawSolution()


def q21():
    k = int(input("Entrez une valeur pour K: "))
    a = float(input("Entrer une valeur pour alpha: "))
    
    model = AffectationUniteSoinLocation(k, a)
    model.solve()
    model.evaluateObjective()
    model.drawSolution()

def q22():
    k = int(input("Entrez une valeur pour K: "))
    a = float(input("Entrer une valeur pour alpha: "))
    
    model = AffectationUniteSoinMaxDist(k, a)
    model.solve()
    model.evaluateObjective()
    model.drawSolution()

def q3():
    
    P = np.array(list(map(int, input("Entrez les valeurs de P: ").split())))
    idx_villes = list(map(int, input("Entrez les index des villes: ").split()))#[5, 8, 10, 13, 14]
    idx_secteurs = list(map(int, input("Entrez les index des secteurs: ").split()))#[5, 8, 10, 13, 14]
                 
                 
    flow(idx_villes, idx_secteurs, P)
    



menu = [("Lancer tout les unittest", lancer_tests),
		("Question 1.2 (affectation secteur)", q1),
		("Question 2.1 (localisation minimiser la moyenne)", q21),
		("Question 2.2 (localisation minimiser le max)", q22),
		("Question 3 (flot max cout min)", q3)]


###Menu
while True:

	os.system('cls' if os.name=='nt' else 'clear')

	print("-1: Stop")

	for i, (name, _) in enumerate(menu):
		print(f"{i}: {name}")

	selected = input("Please enter your choice: ")

	try:
		selected = int(selected)

	except ValueError:
		print(f"{selected} is not a number.")
		continue

	if selected == -1:
		break

	else:

		if selected < 0 or selected >= len(menu):
			print(f"{selected} is an invalid choice.")
			continue
		
		os.system('cls' if os.name=='nt' else 'clear')
		menu[selected][1]()
		input("press any key to continue ....")