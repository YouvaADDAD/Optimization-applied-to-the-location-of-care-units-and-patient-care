/*Exercice 01*/

/*ici dans l'exercice 1 on cherche
une Représentation préfixe en Prolog
des connaissances de l'exercice 2 du TD4  */

/*TBOX*/
subs(chat,felin).
subs(lion,felin).
subs(chien,canide).
subs(canide,chien).
subs(souris,mammifere).
subs(felin,mammifere).
subs(canide,mammifere).
subs(mammifere,animal).
subs(canari,animal).
subs(animal,etreVivant).
subs(and(animal,plante),nothing).
subs(and(animal,some(aMaitre)),pet).
subs(pet,some(aMaitre)).
subs(some(aMaitre),all(aMaitre,humain)).
subs(chihuahua,and(pet,chien)).
subs(lion,carnivoreExc).
subs(carnivoreExc,predateur).
subs(animal,some(mange)).
subs(and(all(mange,nothing),some(mange)),nothing).
equiv(carnivoreExc,all(mange,animal)).
equiv(herbivoreExc,all(mange,plante)).

/*ABOX*/

inst(felix,chat).
inst(pierre,humain).
inst(princesse,chihuahua).
inst(marie,humain).
inst(jerry,souris).
inst(titi,canari).
instR(felix,aMaitre,pierre).
instR(princesse,aMaitre,marie).
instR(felix,mange,jerry).
instR(felix,mange,titi).


/*Exercice 02*/

subsS1(C,C).
subsS1(C,D):- subs(C,D) , C\==D.
subsS1(C,D):- subs(C,E),subsS1(E,D).


/*1)
   1*la régle (1) subsS1(C,C) traduit la reflexivité , C est subsumé par C
   2*la régle (2) dit si C est subsumé par D factuellement et que C different de D, alors C est subsumé structurellement avec D
   ici on quelque sorte on fait le lien entre la subsomption factuelle et structurelle
   3*la régle (3) traduit la transitivité si C est subsumé par E et si E subsumé structurellement par D alors C est subsumé structurellement par D


*/
/*2 Tester
    ?- subsS1(canari,animal).
    true ;
    true ;
    false.

Le premier true c'est a cause de subs(canari,animal)
Le deuxieme true a cause de subs(canari,animal) et subsS1(animal,animal)

    ?- subsS1(chat,etreVivant).
    true ;
    true ;
    false.

Comme un chat est un felin ,les felins sont des mammifere et les mamifere sont des animaux , et animal est subsumée par etreVivant alors chat est subsumé par etreVivant.
On explique les multiples trues qui apparaissent ,c'est du au fait que prolog test toute les possibilité
les deux trues meme raisonnement que precedement.

    ?- subsS1(chien,souris).
    ERROR: Stack limit (1.0Gb) exceeded
    ERROR:   Stack sizes: local: 0.9Gb, global: 0.3Mb, trail: 64.0Mb
    ERROR:   Stack depth: 8,382,937, last-call: 50%, Choice points: 4,191,467
    ERROR:   Possible non-terminating recursion:
    ERROR:     [8,382,936] user:subsS1(chien, souris)
    ERROR:     [8,382,935] user:subsS1(canide, souris)

Trace de subsS1(chien,souris):
    [trace]  ?- subsS1(chien, souris).
	Call: (8) subsS1(chien, souris) ? creep
	Call: (9) subs(chien, souris) ? creep
	Fail: (9) subs(chien, souris) ? creep
	Redo: (8) subsS1(chien, souris) ? creep
	Call: (9) subs(chien, _7328) ? creep
	Exit: (9) subs(chien, canide) ? creep
	Call: (9) subsS1(canide, souris) ? creep
	Call: (10) subs(canide, souris) ? creep
	Fail: (10) subs(canide, souris) ? creep
	Redo: (9) subsS1(canide, souris) ? creep
	Call: (10) subs(canide, _7328) ? creep
	Exit: (10) subs(canide, chien) ? creep
	Call: (10) subsS1(chien, souris) ? creep
	Call: (11) subs(chien, souris) ? abort


On constate ici qu'on a un cycle ,une boucle infini cella est du au fait que prolog ne reconnait pas subs(chien,souris) comme un fait a cause des deux subsumption subs(chien,canide) et subs(canide,chien) un appel infini a subsS1

    ?- subs(souris,some(mange)).
    false.

    ?- subsS1(canari,and(etreVivant,some(mange))).
    false.

    ?- subsS1(chihuahua,canide).
    false.


    ?- subsS1(canari,and(etreVivant,some(mange))).
    false.

    ?- subsS1(chihuahua,canide).
    false.

    ?- subs(and(chien,some(aMaitre)),pet).
    false.

    ?- subs(lion,all(mange,animal)).
    false.

    ?- subs(lion,all(mange,etreVivant)).
    false.

    ?- subs(all(mange,canari),carnivoreExc).
    false.

    ?- subs(and(carnivoreExc,herbivoreExc),all(mange,nothing)).
    false.

    ?- subs(and(and(carnivoreExc,animal),herbivoreExc),nothing).
    false.

A ce stade on a pas encore implementé les autres régles .
*/

subsS(C,D) :- subsS(C,D,[C]).
subsS(C,C,_).
subsS(C,D,_):-subs(C,D),C\==D.
subsS(C,D,L):-subs(C,E),not(member(E,L)),subsS(E,D,[E|L]),E\==D.

/*4

        ?- subsS(canari,animal).
        true ;
        false.

        ?- subsS(chat,etreVivant).
        true ;
        false.

        ?- subsS(chien,canide).
        true ;
        false.

        ?- subsS(chien,chien).
        true ;
        true ;
        false.

    On renvois toujours autant de true qu'il ya d'unification
*/


/*5

        ?- subsS(souris,some(mange)).
        true ;
        false.

La requete reussie parce que prolog n'a pas la definition ,il ne sait pas ce que sa represente concretement
<<all>> ,<<and>>, <<equiv>>, <<inst>>, <<instR>>, prolog ne sais pas comment les interprétes, mais syntaxiquement
les reconnaient et les uniefies bien.

*/

/*6
        ?- subsS(chat,humain).
        false.

        ?- subsS(chien,souris).
        false.

Ici les deux testes echoue parce que un chat n'est pas un humain
et surtout on a pas de boucle infinie pour chien et souris qui est du grace a la liste des faits deja visiter
*/

/*7
1) a la main:
    X = chat ; on est subsumé par sois
    X = felin ; par subs(chat,felin)
    X = mammifere ;par subs(felin,mammifere)
    X = animal ;par subs(mammifere ,animal)
    X = etreVivant ;par subs(animal,etreVivant)
    X = some(mange) ;par subs(etreVivant,some(mange))
2) teste
    ?- subsS(chat,X).
    X = chat ;
    X = felin ;
    X = mammifere ;
    X = animal ;
    X = etreVivant ;
    X = some(mange) ;
    false.

Ici la requete nous renvois toute les subsomption de chat par reflexivité,transitivité

1) a la main
    X = mammifere ;parce que tout concept est subsumé par sois
    X = souris ;par subs(souris,mammifere)
    X = felin ;par subs(felin,mammifere)
    X = canide ;par subs(canide,mammifere)
    X = chat ;par subs(chat,felin),subs(felin,mammifere)
    X = lion ;par subs(lion,felin),subs(felin,mammifere)
    X = chien ;par subs(chien,canide),subs(canide,mammifere)
2) teste
    ?- subsS(X,mammifere).
    X = mammifere ;
    X = souris ;
    X = felin ;
    X = canide ;
    X = chat ;
    X = lion ;
    X = chien ;
    false.
Meme chose ici
*/


/*8 discontiguous subs/2 juste pour evité le warning liee au dispatche*/
:- discontiguous subs/2.
subs(A,B) :- equiv(A,B).
subs(B,A) :- equiv(A,B).

/*9
Avant
    ?- subsS(lion,all(mange,animal)).
    false.
Apres
    ?-  subsS(lion,all(mange,animal)).
    true ;
    false.
par equiv(carnivoreExc, all(mange, animal)) et subs(lion,carnivoreExc) prolog a deduit cela
*/

/*10
On a plus interet a dériver la subsomption factuelle que la structurelle ,parce que subs est le cas de base de subsS
pour subs on ne s'interesse qu'au fait existant ,subsS a deduire les autres faits

prenant par exemple :
equiv(x, y).
subs(y, z).

Un appel à subsS(x, z), qui devrait réussir, fait d'abord un appel à subsS(x, z, [x]). L'unification avec la première
règle de subsS/3 n'est pas possible, l'unification avec la deuxième échoue car subs(x, z) est faux, et il reste ainsi
l'unification avec la dernière règle de subsS/3. Le premier littéral de cette clause est subs(x, E).

Si on a crée des subsomptions factuelles à partir de l'équivalence, alors subs(x, E) peut être unifié avec la
clause subs(A, B) :- equiv(A, B) avec A = x et B = E, et ensuite il trouvera B = y. Cette partie de l'appel
réussit et il est facile à voir que le reste de l'appel réussit également.

Si on a crée des subsomptions structurelles à partir de l'équivalence, subs(x, E) ne pourra être unifié avec
aucune clause, et l'appel échoue.
Les règles de subsS, avec les règles de subs rajoutées à la question 8, permettent de répondre à toute requête
atomique si l'on suppose que la T-Box ne contient que des subsomptions ou équivalences entre concepts atomiques.

En effet, dans ce cas, A est subsumé structurellement à B si et seulement si il existe une suite de concepts
atomiques A0, A1, ..., An avec A0 = A, An = B et, pour tout i, Ai est subsumé factuellement ou équivalent à Ai+1.
Comme les nouvelles règles de subs dans la question 8 transforment les équivalences en subsomptions factuelles et
subsS teste l'existence d'un chemin de subsomptions factuelles entre A et B, subsS renverra le bon résultat.
Cela n'est cependant pas le cas si la T-Box contient des subsomptions ou équivalences entre concepts non-atomiques.
Par exemple, si on a dans la T-Box
	subs(x, and(y, z)).
	subs(and(z, y), w).
l'appel à subsS(x, w) échoue, même si x et w sont des concepts atomiques, car, pour qu'il réussisse comme attendu, il faut trouver un moyen de dire à Prolog que and(y, z) et and(z, y) sont la même chose (ce qui est l'objet du prochain exercice).
*/

/*Exercice 03*/

:- discontiguous subsS/3.
subsS(C, and(D1, D2), L) :- D1 \= D2, subsS(C, D1, L), subsS(C, D2, L).
subsS(C, D, L) :- subs(and(D1, D2), D), E=and(D1, D2), not(member(E, L)), subsS(C, E, [E|L]), E\==C.
subsS(and(C, C), D, L) :- nonvar(C), subsS(C, D, [C|L]).
subsS(and(C1, C2), D, L) :- C1 \= C2, subsS(C1, D, [C1|L]).
subsS(and(C1, C2), D, L) :- C1 \= C2, subsS(C2, D, [C2|L]).
subsS(and(C1, C2), D, L) :- subs(C1, E1), E = and(E1, C2), not(member(E, L)), subsS(E, D, [E|L]), E\==D.
subsS(and(C1, C2), D, L) :- Cinv=and(C2, C1), not(member(Cinv, L)), subsS(Cinv, D, [Cinv|L]).

/* 1 les testes:
   1)
    ?-subsS(chihuahua,and(mammifere,some(aMaitre))).
    true.
   2)
    ?- subsS(and(chien, some(aMaitre)), pet).
    true.
   3)
    ?- subsS(chihuahua, and(chien, pet)).
    true.
Toutes ces requetes ont aboutit et renvois true , par contre si on avais mis ";" au lieu de entrer on aurait eu plusieur
True ce qui serai du au fait que prolog est exhaustif

*/


/*2 la situation traité

    1) subsS(chihuahua,and(mammifere,some(aMaitre))):
    pour avoir cette sumbsomption on part de "subs(chihuahua,and(chien,pet))" on aura que subs(chien,mammifere) et
    subs(pet,some(aMaitre)).

    On effet c'est la régle "subsS(C, and(D1, D2), L) :- D1 \= D2, subsS(C, D1, L), subsS(C, D2, L)" a qui on dois le résultat
    cette régle nous dis que  si C est subsumé par D1 et si il est subsumé par D2 alors C est subsumé par (D1 and D2)

    sans cette régle la requete ?-subsS(lion, and(mammifere, predateur)). aurais renvoyer false

    2)la régle subsS(C, D, L) :- subs(and(D1, D2), D), E=and(D1, D2), not(member(E, L)), subsS(C, E, [E|L]), E\==C.
    cette régle nous dis que si (D1 et D2) est subsumé factuellement par D et si C est subsumé structurellement par (D1 et D2) alors C est sumbsumé
    structurellement par D

    typiquement si on avais :
    subs(and(D1,D2),D)
    subs(C,D1)
    subs(C,D2)

    Sans la règle :
	?- subsS(C, D).
	false.

	Avec la règle :
	?- subsS(C, D).
	true .

    3)subsS(and(C, C), D, L) :- nonvar(C), subsS(C, D, [C|L]):
    cette regle traduit que si C est subsumé par D alors (C et C) est subsumé par D , enfaite ici sa traduit le fais
    que and(C,C) etais la meme chose que C (and(C,C)=C). (ici le nonvar(C) stipule que C ce n'est pas une variable ex: nonvar(chihuahua) renvera true
    nonvar(X) is true if and only if X is not a variable.)

    typiquement une requete qui n'aura pas marché : subs(and(chien,chien),chien)

    Sans la règle :
	?- subs(and(chien,chien),chien).
	false.

	Avec la règle :
	?- subs(and(chien,chien),chien).
	true ;
	false.

    4)subsS(and(C1, C2), D, L) :- C1 \= C2, subsS(C1, D, [C1|L]).
      subsS(and(C1, C2), D, L) :- C1 \= C2, subsS(C2, D, [C2|L]).

    ces régles nous disent que  si C1 est subsumé par D alors (C1 et C2) est subsumé par D ou si C2 est
    subsumé par D alors (C1 et C2)
    si on raisonné par ensemble si j'ai A ⊂ C alors A ∩ B ⊂ C (vu que ici on a x appartient a A donc x appartient
    a C ou sinon l'ensemble vide apparient a C)

    Sans ces deux règles :
	?- subsS(and(animal, predateur), etreVivant).
	false.

	Avec une seule des deux règles (mode exhaustif) :
	?- subsS(and(animal, predateur), etreVivant).
	true.

    5)subsS(and(C1, C2), D, L) :- subs(C1, E1), E = and(E1, C2), not(member(E, L)), subsS(E, D, [E|L]), E\==D:
    Ici la régle dis que ci C1 est subsumé factuellement par E1 et si (E1 et C2) est subsumé structurellement
    par D alors (C1 et C2) est subsumé structurellement par D

    sans la règle :
	?- subsS(and(X, some(aMaitre)), pet).
	X = animal ;
	false.

	avec la régle:
	?- subsS(and(X, some(aMaitre)), pet).
	X = animal ;
	X = chat ;
	X = lion ;
	X = chien ;
	X = canide ;
	X = souris ;
	X = felin ;
	X = mammifere ;
	X = canari ;
	X = and(animal, some(aMaitre)) ;
	X = chihuahua ;

    6)subsS(and(C1, C2), D, L) :- Cinv=and(C2, C1), not(member(Cinv, L)), subsS(Cinv, D, [Cinv|L]):
    c'est la régle symetrie ou la régle lie a la cummutativité c'est a dire and(C1,C2)=and(C2,C1) . La régle nous dis
    que si (C2 et C1) est subsumé structrellement par D alors (C1 et C2) est lui est aussi subsumé structrullement.

    d'ailleur avec cette régle sa ,elle nous aura permis de reduire une des 2 régle de (4).
    On va imaginé un exemple :
    Imaginant avoir ces fais
    subs(and(A, B), C).
	subs(C, D).
	
	Sans la règle :
	?- subsS(and(B, A), D).
	false.
	
	Avec la règle :
	?- subsS(and(B, A), D).
	true ;
	false.

    subs(and(animal,some(aMaitre)),pet).
    subs(pet,some(aMaitre)).

    avec la regle subsS(and(some(aMaitre),animal),some(aMaitre)). renvrai true
*/

/*3
  Ces régles ne suffisent malheureusement pas pour gérer toute requéte ne contenant que des concepts atomiques
  ou avec intersection vis-a-vis d’une T-Box ne contenant que des subsomptions entre concepts atomiques
  ou avec intersection

  En effet on prenant n'importe quel requete du style subsS(A, and(B, B)) nous retournera false parce que la régle
  (1) ne prend que le and du memebre de droite et pas celui de gauche .

EXAMPLE:
  ?- subsS(mammifere,animal).
  false.

Entre autre biensur
*/


/*Exercice 03*/

/*1) Ajout des régles */
:- discontiguous subs/2.
subs(all(R, C), all(R, D)):-subs(C, D).


/*On aura pu utilisé subsS(all(R, C), all(R, D), L):-subsS(C, D, L) mais on aura pas traité tout les cas
Exemple: subsS(all(mange, canari), carnivoreExc) renvois false alors quel aurait du renvoye true.

donc en rajouté plutot une sumbsoption factuelle c'est en quelque sorte directement un cas de base
*/


/*2) les testes

	?- subsS(lion, all(mange, etreVivant)).
	true ;
	false.


	?- subsS(all(mange, canari), carnivoreExc).
	true ;
	false.
*/


/*3) Encore d'autre testes

1)
?- subsS(and(carnivoreExc,herbivoreExc),all(mange,nothing)).
false.

2)
?- subsS(and(and(carnivoreExc, herbivoreExc), animal), nothing).
false.

les requetes ne reussissent parce que prolog ne sais pas passé de "subsS(and(animal, plante), nothing)" à
subsS(all(mange, and(animal, plante)), all(mange, nothing)).
et	subsS(and(carnivoreExc, herbivoreExc), all(mange, and(animal, plante))) a partir de
subsS(carnivoreExc, all(mange, animal)) et subsS(herbivoreExc, all(mange, plante))
*/


:- discontiguous subsS/3.
/*1*/ subsS(C, all(R, and(D1, D2)), L) :- D1 \= D2, subsS(C, all(R, D1), L), subsS(C, all(R, D2), L).
/*2*/ subsS(C, all(R, D), L) :- subs(and(D1, D2), D), E = all(R, and(D1, D2)), not(member(E, L)), subsS(C, E, [E|L]), E\==C.

/*
la régle 1: si C est subsumé structurellement par all(R,D1) et que C est subsumé par all(R,D2) alors C est subsumé par
all(R,and(D1,D2)).

la régle 2:si (D1 et D2) est subsumé par D et si C est subsumé par all(R,(D1 et D2)) alors C est subsumé par all(R,D).


?- subsS(and(carnivoreExc,herbivoreExc),all(mange,nothing)).
true .

?- subsS(and(and(carnivoreExc, herbivoreExc), animal), nothing).
true .

?- subsS(and(and(carnivoreExc, animal), herbivoreExc), nothing).
true .

*/


/*4 Ajout de régle */
:- discontiguous subs/2.
subs(all(R, I), all(R, C)):-inst(I, C).

/*Si I est une instance de C alors pour tout (R,I) est subsumé par pour tout (R,C)

?- subsS(all(mange, titi), all(mange, canari)).
	true ;
	false.

?- subsS(all(mange, titi), all(mange, etreVivant)).
	true ;
	false.

?- subsS(all(aMaitre, pierre), all(aMaitre, humain)).
	true ;
	false.

*/


/*5
non il n'est pas necessaire d'ecrire cette régle ,on effet dans FL  some(Role) et non some(Role, concept).

*/

/*6


la requete subsS(lion,X) renvois on précedant de la meme maniere que dans l'exercice 2.7 a ceci prés qu'on aura les
    X = lion ;subsS(lion,lion) lion est subsumé par sois
    X = felin ;subs(lion,felin)
    X = carnivoreExc ;subs(lion,carnivoreExc)
    X = mammifere ;subs(felin,mammifere)
    X = animal ;subs(mammifere,animal)
    X = etreVivant ;subs(animal,etreVivant)
    X = some(mange) ;subs(animal,some(mange))
    X = predateur ;subs(carnivoreExc,predateur)
    X = all(mange, animal) ;equiv(carnivoreExc,all(mange,animal)) (double subsomption)
	X = all(mange, etreVivant); par subs(animal,etreVivant)
	X = all(mange, some(mange));par subs(animal,some(mange)).

Test prolog:
    ?- subsS(lion,X).
    X = lion ;
    X = felin ;
    X = carnivoreExc ;
    X = mammifere ;
    X = animal ;
    X = etreVivant ;
    X = some(mange) ;
    X = predateur ;
    X = all(mange, animal) ;
    X = all(mange, etreVivant) ;
    X = all(mange, some(mange)) ;
    false.

la requete subsS(X,predateur) a la main donne:
    X=predateur ; par subsS(predateur,predateur)
    X=carnivoreExc; par subs(carnivoreExc,predateur)
    X=lion; par subs(lion ,carnivoreExc)
	X = all(mange, animal);par equiv(carnivoreExc,all(mange,animal)) (double subsomption)
	X = all(mange, mammifere);par subs(mammifere,animal)
	X = all(mange, canari); par subs(canari,animal)
	X = all(mange, souris);par subs(souris,mammifere)
	X = all(mange, felin);par subs(felin,mammifere)
	X = all(mange, canide);par subs(canide,mammifere)
	X = all(mange, chat);par subs(char,felin)
	X = all(mange, lion);par subs(lion,felin)
	X = all(mange, chien);par subs(chien,canide)

Test prolog:
    ?- subsS(X,predateur).
    X = predateur ;
    X = carnivoreExc ;
    X = lion ;
    X = all(mange, animal) ;
    X = all(mange, chat) ;
    X = all(mange, lion) ;
    X = all(mange, chien) ;
    X = all(mange, souris) ;
    X = all(mange, felin) ;
    X = all(mange, canide) ;
    X = all(mange, mammifere) ;
    X = all(mange, canari) ;

    on constate une boucle infinie
*/


/*Exercice 05*/

/*
Un  ensemble de régles ainsi produit est complet pour un langage donné s’il peut prouver toute subsomption C est
subsumé par D correcte à partir du moment ou tous les termes de la T-Box et de la requete appartiennent au
langage donné. donc en conclusion cet ensemble n'est pas FL- .
Exemple:
    ?- subsS(lion,and(carnivoreExc,carnivoreExc)).
    false.
*/



/*Exercice 06*/

instS(I, C):-inst(I, D), subsS(D, C).

/*
Si I est une instance de D et si D est subsumé strcuturellement par C alors I une instance structurelle de C
toute en gardant les régles liee a la subsomption structurelle
*/


/*2 les testes

    ?- instS(felix,mammifere).
    true .

   par inst(felix,chat) et subs(chat,felin) et subs(felin,mammifere).

   ?- instS(princesse,pet).
   true .

    par inst(princesse,chihuahua) et subs(chihuahua,and(pet,chien)) donc subs(chihuahua,pet) et subs(chihuahua,chien)

*/

/*3 Ajout de régles*/

instS(I, some(R)):- instR(I, R, _).

/*si I est une instance de R aloes I est une instance de some(R)*/

/*les testes

    ?- instS(felix, some(mange)).
    true .

    ?- instS(princesse,some(aMaitre)).
    true .

    ?- instS(felix,some(aMaitre)).
    true .


*/

/*4 definir contreExAll(I,R,C) */


contreExAll(I, R, C):- instR(I, R, I2), not(instS(I2, C)).

:-discontiguous instS/2.
instS(I, all(R, C)) :- not(contreExAll(I, R, C)).


/*Quelque testes

    ?- instS(felix,all(mange,animal)).
    true.

    ?- instS(titi,all(mange,personne)).
    true.

    ?- instS(felix,all(mange,mammifere)).
    false.


*/

/*5
Ici on a pas besoin de le faire on effet parce que si I est une instance structuelle de C on pourra applique toute
les subsomptions structurelle liee a C

Exemple:
    ?- instS(felix,X).
    X = chat ;
    X = felin ;
    X = mammifere ;
    X = animal ;
    X = etreVivant ;
    X = some(mange) ;
    X = some(aMaitre) ;
    X = some(mange) ;
    X = some(mange) ;
    X = all(_322976, _322978).
*/


/*6
    ?- instS(felix,pet).
    false.

    ?- instS(felix,carnivoreExc).
    false.

Il manque ici un moyen pour dire a prolog que

    instS(felix, all(mange, animal))
	instS(felix, some(aMaitre))
	instS(felix, animal)
sont vrai

ce qui manquerai c'est une régle de transitivité comme celle de subsS et les regles du "and" typiquement traduire
les régles de subsS en instS (sa serai plus une intuition )

*/






