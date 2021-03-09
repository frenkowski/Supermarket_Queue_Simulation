# Simulazione e Confronto tra due Tipologie di Coda al Supermercato

## Team
Francesco Stranieri (<frenkowski+github@gmail.com>)  
Davide Marchetti (<dvdmarchetti663+kaggle@gmail.com>)  
Mattia Vincenzi (<vincenzi.mattia97+github@gmail.com>)  

## Abstract
Questo progetto si pone l'obiettivo di studiare, e confrontare, le differenze tra le tipologie di code quotidianamente utilizzate all'interno dei supermercati [[1]](#1).
In Italia, sono per lo più diffuse due diverse filosofie, come mostrato in Figura 1: ![Figura 1](https://www.frenkowski.it/wp-content/uploads/2020/08/mesa.png)
:
- **Classic** (1a): rappresenta la modalità più classica; è quindi presente un numero di code pari al numero di casse aperte in quel preciso momento. 
- **Snake** (1b): rappresenta invece la modalità 'a serpentone'; si compone di un unico ingresso per l'accesso ad una coda comune, la quale porta all'uscita che permette il raggiungimento delle casse. 

Gli elementi chiave utilizzati per confrontare le diverse tipologie di code sono: *il numero totale, e medio, di persone in coda per ogni istante temporale*, *il tempo medio di attesa in coda*, *il tempo medio di permanenza* e il *comportamento delle persone all'interno del supermercato*. 

Per modellare lo scenario oggetto di studio è stato utilizzato il framework open-source MESA [[2]](#2), sviluppato in Python.

## References
<a id="1">[1]</a> 
Daichi Yanagisawa, Yushi Suma, Akiyasu Tomoeda, Ayako Miura,Kazumichi Ohtsuka, and Katsuhiro Nishinari. 
Walking-distance introduced queueing model for pedestrian queueing system: Theoretical analysis and experimental verification. 
Transportation Research Part C: Emerging Technologies, 37:238 – 259, 2013.

<a id="2">[2]</a> 
Mesa: Agent-based modeling in python 3+.
https://mesa.readthedocs.io/en/master/
