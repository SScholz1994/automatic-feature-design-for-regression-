from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
import numpy as np
from scipy.optimize import curve_fit
import pylab as py 



# LEAVE ONE OUT 6 FOLD CROSS VALIDATION zu Galileos Rampenexperiment

# Daten bzw. Zeiten
null = 0
viert = np.average(np.loadtxt('Experimentdaten', usecols=(1)))
halb = np.average(np.loadtxt('Experimentdaten', usecols=(2)))
zwDrit = np.average(np.loadtxt('Experimentdaten', usecols=(3)))
drViert = np.average(np.loadtxt('Experimentdaten', usecols=(4)))
ganz = np.average(np.loadtxt('Experimentdaten', usecols=(5)))

X = np.array([null, viert, halb, zwDrit, drViert, ganz])
Y = np.array([0, 0.25, 0.5, 0.67, 0.75, 1]) 				# Positionen auf der Rampe

print X
print 

# zur Visualisierung der Anfangsbedingungen
plt.figure()
plt.plot(X,Y, 'o')
plt.xlabel('Input: die Zeitwerte x')
plt.ylabel('Output: die zuruckgelegte Strecke y')	
plt.title('gegeben: Durchschnittliche Datenpunktpaare')
plt.show()


# 3 FUNKTIONEN: weights, Fit, ERROR


#Bestimmung der Gewichte (der features)
def weights(X, Y):							
	M = np.zeros((6,6))											# k = P = 6 Punkte haben wir, also testen wir bis zum Polynom 6. Grades maximal. 
	N = np.zeros((6,6))
	o = np.zeros((6))
	
	for p in range(0, len(X)):
		x = X[p]
		f_p = np.array([x, x**2, x**3, x**4, x**5, x**6])		#einfacher Vektor fuer jedes x: fixed bases of poynomial features


		for j in range(0,6):									# Berechnung aller Eintraege der 6x6 Matrix: Spalten * Zeilenvektor  (linke Seite)
			for k in range(0,6):
				M[j][k] = f_p[j]*f_p[k] 



		M = M+ N 												# Addition der 6 maximalen Matrizen 
		N = np.copy(M)		

		y = Y[p]
		fXy = f_p * np.array([y,y,y,y,y,y]) 					# Multiplikation fuer die rechte Seite d. Gleichung: 
																# fi * y, wobei y: Original Outputs
		fXy = fXy + o 											# Addition der 5 Vektoren
		o = np.copy(fXy)

	w6 = np.linalg.solve(M, fXy)								# Loesung des Gleichungssystems summe(M)*w = summe(f_i*y) fuer Merkmale M=1 bis M=6
	w5 = np.linalg.solve(M[:5,:5], fXy[:5])
	w4 = np.linalg.solve(M[:4,:4], fXy[:4])
	w3 = np.linalg.solve(M[:3,:3], fXy[:3])
	w2 = np.linalg.solve(M[:2,:2], fXy[:2])
	w1 = np.linalg.solve(M[:1,:1], fXy[:1])

	w = np.array([w1, w2, w3, w4, w5, w6])						# matrix aller gewichte vom ausgewaehlten Trainingsset


	W = np.zeros((6,6))											# lediglich fuer einfachere Rechnung spaeter: w-Vektoren mit 0en auffuellen
	for j in range(0,6):
		W[j][:j+1] = w[j]
	return W									




def Fit(W,x):  # W: hier: W[i] --> eine Zeile der W Matrix   															 																			 																						  
	return 	W[0]*x + W[1]*x**2 + W[2]*x**3 + W[3]*x**4 + W[4]*x**5 + W[5]*x**6	 # W fuer weights eines bestimmten Merkmalsgrads, die die endgueltige Funktion dafuer bestimmen.



def ERROR(X,Y, w):   											# mittlerer quadratischer Fehler				
																# x und y: entweder Training Sets oder Testing Sets, w: Gewichte bis entsprechendes Merkmal
	Sum = np.zeros(6)
	for p in range(0,len(X)):  									# fuer alle x werte in der Menge: wie gut passt das hier?
		x = X[p]
		y = Y[p]
		f_p = np.array([x, x**2, x**3, x**4, x**5, x**6])  		# maximaler Merkmalsvektor
														    	# Mermale bis entsprechendem M. 
		Sum[p] = (np.sum(f_p * w) - y)**2 			
		
		err = np.divide(1.,len(X)) * np.sum(Sum)	
	return err

										
	


#zur CROSS VALIDATION


# Zwischenziel: Ermittlung der Testing errors 
# --> Initialisierung der Arrays
allErrors_test = np.zeros((6,6)) 		
allErrors_train = np.zeros((6,6))						
avgErrors_Test = np.zeros(6)
avgErrors_Train = np.zeros(6)

# fuer kontinuierlichen Plot der gefundenen Funktion
x_cont = np.linspace(0, X[len(X)-1], 100)

#partitions: 
# --> 6-fold cross validation 

# bei groesseren Mengen vielleicht mit Modulo einteilen oder einer integrierten random Funktion. 
# Hier:  jeder Punkt ist einmal Testmenge. Daher wird je ein Punkt wird mit delete entfernt

			
for k in range(0,len(X)):

	Testx = np.array([X[k]])
	Testy = np.array([Y[k]])
	TrainX = np.delete(X, k)
	TrainY = np.delete(Y, k)

	#Die berechneten Gewichte: 
	
	W = weights(TrainX, TrainY)			# Bereichnnung der Gewichte in der Funktion	fuer alle Merkmale
										# wobei w0 = b = 0, so viel kann man schliessen
	
	TrainErr = np.zeros(6)
	TestErr = np.zeros(6)


	for m in range(0,6):
		TrainErr[m] = ERROR(TrainX, TrainY, W[m])  

		TestErr[m] = ERROR(Testx, Testy, W[m])

		
		plt.figure()
		plt.plot(x_cont,Fit(W[m], x_cont) )       											# Plot der gelernten Funktion
		plt.plot(TrainX, TrainY, 'go')														# Trainingspunkte
		plt.plot(Testx, Testy, 'ro')														# Testpunktepaar
		plt.xlabel('Input: die Zeitwerte x')
		plt.ylabel('Output: die zuruckgelegte Strecke')													
		plt.title('Training Set ohne Punkt P =' +str(k+1) +', bis Mermal M = '+str(m+1))
		#plt.show()
		

	D = np.array([1,2,3,4,5,6])	 # D = degree: fuer die korrekte Achseneinteilung

	#Fehlerplots fuer das gesamte Set k
	plt.figure()								
	plt.plot(D, TrainErr, 'go-')
	plt.plot(D, TestErr, 'ro-')	
	plt.legend(('Trainingsfehler', 'Testfehler'))	
	plt.xlabel('D  (degree)')
	plt.ylabel('Fehler')
	plt.title('Mittlere quadr. Fehler bei Test mit Punkt P = '+str(k+1))
 

	# Speicherung der Fehlerwerte
	allErrors_test[k] = TestErr 									  # Matrizen (6x6), die am Anfang initialisiert werden

	allErrors_train[k] = TrainErr

	print 'Grad des Fehlerminimums des ' + str(k+1)+ '-ten Testsests:'
	print np.argmin(allErrors_test[k])+1
	print 


	# Abschuss k- Schleife
#plt.show()

for a in range(0,6):												   # avgErrors oben initialisiert (Array,(6))
	avgErrors_Test[a] = np.average(allErrors_test[:6,a:a+1])	       # Durchschnitt aus einer gesamten Spalte jeweils, 
																	   # ueber alle Zeilen   :6
	avgErrors_Train[a] = np.average(allErrors_train[:6,a:a+1])

	


plt.figure()
plt.plot(D, avgErrors_Test, 'ro-')
plt.plot(D, avgErrors_Train, 'go-')
plt.xlabel('Grad D des Polynoms')
plt.ylabel('mittlere Fehlerwerte')
plt.legend(('Testfehler', 'Trainingsfehler'))	
plt.title('Durchschnittlicher Testfehler der Merkmale D = 1 bis 6')
#plt.show()

#print avgErrors 
Dtop = np.argmin(avgErrors_Test) 										# index + 1 ist gesuchter Degree D


# Final: wir formen den Merkmalsvektor (feature vector) fuer alle Punkte imgesamten Datenset
# 		 und loesen das Least Square Problem ueber dieses gesamte Datenset, um das finale Modell zu finden.
#		 Dabei ist nur der Eintrag zu Dtopmoch relevant

Wtop = weights(X, Y)[Dtop] 

print 
print 
print 
print '------------------------------------------------------------------------------------------'
print 'Final automatisch gefundene Funktion:'
print 'f(x) = ' +str(Wtop[0])+'* x + ' + str(Wtop[1])+'* x^2 + ' + str(Wtop[2]) + '* x^3 + ' +str(Wtop[3])+'* x^4 + '+str(Wtop[4])+'* x^5 + '+str(Wtop[5])+'* x^6'


plt.figure()
plt.plot(X,Y,'go')
plt.plot(x_cont, Fit(Wtop,x_cont),'r')
plt.xlabel('Input: die Zeitwerte x')
plt.ylabel('Output: die zuruckgelegte Strecke y')
plt.legend(('Experiementdaten', 'Funktion'))
plt.title('Endergebnis: Poynom mit gelernten optimalen Gewichte zum gesamten Datensatz')
plt.show()
		








	
	
	
	


	
	







