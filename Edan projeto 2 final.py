import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#def getlin(x, a, b, c):  # x = VGS; a = K; b = Vt; c = n
#    return np.piecewise(x,[x < b, x >= b],[lambda x: 0, lambda x: a * np.power(np.maximum(x - b, 0), c)])

def getlin_ID_GM(x, a, b):  # x = VGS; a = Vt; b = m;
    return np.piecewise(x,[x-a < 0, x-a >= 0],[lambda x: 0, lambda x: (x-a)/b])

#a) e b)
plt.close('all')
dfo = pd.read_csv("C:/E2/EDAN/P1G4_W50L5_Output.csv",skiprows=256, usecols=[0,1,2,3])
dfo_VG = dfo.iloc[:,2].values
dfo_VD = dfo.iloc[:,1].values
dfo_ID = dfo.iloc[:,3].values
df_sat= pd.read_csv("C:/E2/EDAN/P1G4_W50L5_Transfer_saturation.csv",skiprows=251, usecols=[0,1,2,3])
dfVG_sat = df_sat.iloc[:,1].values
dfVD_sat = df_sat.iloc[:,2].values
dfID_sat = df_sat.iloc[:,3].values
#dfGM_sat = df_sat.iloc[:,4].values
#c)
index = np.argmax(dfVG_sat)
dfVG_sat2 = dfVG_sat[: index+1]
dfVD_sat2 = dfVD_sat[: index+1]
dfID_sat2 = dfID_sat[: index+1]
#dfGM_sat2 = dfGM_sat[: index+1]

# Part1
#1. Função caracteristica do TFT em saturação
plt.figure(1)
plt.plot(dfVG_sat2, 1000 * dfID_sat2, linestyle='None',marker='.',color='b',label="ID(Vgs)")
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA])")
plt.legend()  
plt.grid(True) 
plt.show()

#2.
#2.1 e 2.2 FOTO
#2.3
####calcular gm com derivade de id em fun cao a VG:  dId/dVG=gm
gm =[]
largura = (len(dfID_sat2)-1)
for i in range(largura):
    gm.append((dfID_sat2[i+1]-dfID_sat2[i])/(dfVG_sat2[i+1] - dfVG_sat2[i]))       
    
#vetor gm
plt.figure(2)
plt.plot(dfVG_sat2[: largura], gm, linestyle='None',marker='.',color='b', label='vetor numérico de gm')
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("gm [Siemes]")
plt.legend()  
plt.grid(True)  
plt.show()
plt.figure(3)
id_gm = dfID_sat2[: largura]/gm
plt.plot(dfVG_sat2[: largura], id_gm, linestyle='None',marker='.',color='b', label='vetor numérico de Id/gm')
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Id/gm [A/Siemes]")
plt.legend()  
plt.grid(True)  
plt.show()
#2.4
#com o gm/ID=(VGS-Vt)/m, sabemos gm, ID e Vgs. calculamos entao VT e m.
plt.figure(4)
[Vt, m], _ = curve_fit(getlin_ID_GM, dfVG_sat2[: largura].flatten(), id_gm)
new2 = getlin_ID_GM(dfVG_sat2[: largura], Vt, m)

plt.plot(dfVG_sat2[: largura], id_gm, linestyle='None',marker='.', label="curva ID/gm")
plt.plot(dfVG_sat2[: largura], new2, label="curva aproximada",)     
plt.title("")
plt.xlabel("Vgs[V]")
plt.ylabel("ID/gm [A/Siemes]")
plt.legend()  
plt.grid(True) 
plt.show()
print("Vt =", Vt)
print("m  =", m)

#2.5
def getlin_k(x, a):  # x = VGS; a = K;
    return np.piecewise(x,[x-Vt < 0, x-Vt >= 0],[lambda x: 0, lambda x: a * np.power(np.maximum(x - Vt, 0), m)])

plt.figure(5)
[K], _ = curve_fit(getlin_k,dfVG_sat2[: largura].flatten(), dfID_sat2[: largura].flatten())
new3 = getlin_k(dfVG_sat2[: largura], K)
plt.plot(dfVG_sat2[: largura],1000* dfID_sat2[: largura], label="curva real ID(Vgs)",  linestyle='None',marker='.')
plt.plot(dfVG_sat2[: largura], 1000*new3, label="curva aproximada ID(Vgs)")     
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA]")
plt.legend()  
plt.grid(True) 
plt.show()
print("K =", K)

# Part2
#2. Função caracteristica do TFT em saturação
#2.1 e 2.2 achar lambda para diferentes VGS [1 ... 7]
lambda_s = []
VG_values = []

plt.figure(6)
for VG_val in np.unique(dfo_VG[dfo_VG > 0]):
    indices = np.where(dfo_VG == VG_val)[0]  
    dfo_VD_subset = dfo_VD[indices]  
    dfo_ID_subset = dfo_ID[indices]  
    
    plt.plot(dfo_VD_subset, 1000 * dfo_ID_subset, label=f"ID(VDS) at VG={VG_val}", linestyle='None', marker='.')
    plt.xlabel("Voltage VDS (V)")
    plt.ylabel("Current IDS (mA)")
    plt.legend()
    plt.grid(True)
    
    VDS2 = dfo_VD_subset[-1]
    ID2 = dfo_ID_subset[-1]
    VDS1 = dfo_VD_subset[-21]
    ID1 = dfo_ID_subset[-21]
    lambda_value = (ID2 - ID1) / (ID1*VDS2 - ID2*VDS1)
    lambda_s.append(lambda_value)
    VG_values.append(VG_val)
    print(f"Lambda for VG={VG_val}: {lambda_value}")    
plt.show()

#2.3 plot lambda(VGS)
plt.figure(7)
plt.plot(VG_values, lambda_s, label="Lambda(VGS)", linestyle='None', marker='.')
plt.xlabel("Voltage VGS (V)")
plt.ylabel("Lambda (1/V)")
plt.legend()
plt.grid(True)
plt.show()

#2.4 e 2.5 curve fit de lambda
degree = 4
coefficients = np.polyfit(VG_values, lambda_s, degree)  #define coeficentes f(vgs)= Ax^2 + Bx + C ...
poly_fit = np.poly1d(coefficients)                      #define o polinomio a partir de coeficentes para poder colocar f(x)

# Temos o polinomio:> poly_fit e vamos criar valores de VG min ate VG max para fazer o plot
VG_fit = np.linspace(min(VG_values), max(VG_values), 100)   # 100 pontos para o ploting
lambda_fit = poly_fit(VG_fit)                               # valores do lambda para os pontos vg criados

plt.figure(8)
plt.plot(VG_values, lambda_s, '.', label="Original Data")
plt.plot(VG_fit, lambda_fit, '-', label=f"Polynomial Fit (degree={degree})")
plt.xlabel("Voltage VGS [V]")
plt.ylabel("Lambda [1/V]")
plt.legend()
plt.grid(True)
plt.show()

# # definir K com o polinomio de lambda
def getlin_k2(x, a):  # x = VGS; a = K;
    return np.piecewise(x,[x-Vt < 0, x-Vt >= 0],[lambda x: 0, lambda x: a * np.power(np.maximum(x - Vt, 0), m)*(1+ (7*poly_fit(x)))])
    #return np.piecewise(x,[x-Vt < 0, x-Vt >= 0],[lambda x: 0, lambda x: a * np.power(np.maximum(x - Vt, 0), m)*(1+ (7*0.2))])

plt.figure(9)
[K2], _ = curve_fit(getlin_k2,dfVG_sat2[: largura].flatten(), dfID_sat2[: largura].flatten(), bounds=(0,1))
new4 = getlin_k2(dfVG_sat2[: largura], K2)
plt.plot(dfVG_sat2[: largura],1000* dfID_sat2[: largura], label="curva real ID(Vgs)",  linestyle='None',marker='.')
plt.plot(dfVG_sat2[: largura], 1000*new4, label="curva aproximada ID(Vgs)")     
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA]")
plt.legend()  
plt.grid(True) 
plt.show()
print("K =", K2)

# Part3
#3. Com	base na característica de saída, para VGS máximo
#3.1
# obter Vgs max e os vetores de VGS max = 6
VGS_max = 6 
indices = np.where(dfo_VG == VG_val)[0] 
VDS_max = dfo_VD[indices]
ID_max = dfo_ID[indices]

#funcao id linear especial one
def ID_model(x, a): # x = VDS, a = alpha        poly_fit(VGS_max)       ((VGS_max - Vt)**0) 
    return np.piecewise(x,[x < a*(VGS_max-Vt), x >= a*(VGS_max-Vt)],
                        [lambda x: 2*(K2/a)*((VGS_max - Vt)**(m-2))*(VGS_max - Vt - (x / (2 * a))) * x * (1 + ( poly_fit(VGS_max)    * x)), 
                          lambda x: (K2)*((VGS_max - Vt)**(m))*(1 + ( poly_fit(VGS_max)    * x))])
plt.figure(10)
[alpha], _ = curve_fit(ID_model, VDS_max, ID_max,bounds=([0], [2]))
new5 = ID_model(VDS_max, alpha)
plt.plot(VDS_max, 1000* ID_max, label="curva real ID(Vgs)",  linestyle='None',marker='.')
plt.plot(VDS_max, 1000*new5, label="curva aproximada ID(Vgs)")     
plt.title("")
plt.xlabel("Voltage VDS [V]")
plt.ylabel("Current ID [mA]")
plt.legend()  
plt.grid(True) 
plt.show()
print(f"Para VGS max = {VGS_max}:")
print("alpha =", alpha)
#print("gama =", m-2)

#########otimização

def getlin_otim(x, a, b, c):  # x = VGS; a = K, b = Vt, c = M;
    return  a * np.power(np.maximum(x - b, 0), c)*(1 + ( poly_fit(x) * 7))

plt.figure(11)
[Ko,Vto,mo], _ = curve_fit(getlin_otim,dfVG_sat2[: largura].flatten(), dfID_sat2[: largura].flatten(),bounds=([0,-3,1], [1,1,4]))
new6 = getlin_otim(dfVG_sat2[: largura], Ko,Vto,mo)
plt.plot(dfVG_sat2[: largura],1000* dfID_sat2[: largura], label="curva real ID(Vgs)",  linestyle='None',marker='.')
plt.plot(dfVG_sat2[: largura], 1000*new6, label="curva aproximada ID(Vgs)")     
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA]")
plt.legend()  
plt.grid(True) 
plt.show()
print("K0 =", Ko)
print("VTO =", Vto)
print("Mo =", mo)

VGS_max = 6 
indices = np.where(dfo_VG == VGS_max)[0] 
VDS_max = dfo_VD[indices]
ID_max = dfo_ID[indices]

#funcao id linear especial one
def ID_model(x, a): # x = VDS, a = alpha        poly_fit(VGS_max)       ((VGS_max - Vt)**0) 
    return np.piecewise(x,[x < a*(VGS_max-Vto), x >= a*(VGS_max-Vto)],
                        [lambda x: 2*(Ko/a)*((VGS_max - Vto)**(mo-2))*(VGS_max - Vto - (x / (2 * a))) * x * (1 + ( poly_fit(VGS_max)* x)), 
                          lambda x: (Ko)*((VGS_max - Vto)**(mo))*(1 + ( poly_fit(VGS_max)    * x))])
plt.figure(12)
[alpha], _ = curve_fit(ID_model, VDS_max, ID_max,bounds=([0], [2]))
new7 = ID_model(VDS_max, alpha)
plt.plot(VDS_max, 1000* ID_max, label="curva real ID(Vgs)",  linestyle='None',marker='.')
plt.plot(VDS_max, 1000*new7, label="curva aproximada ID(Vgs)")     
plt.title("")
plt.xlabel("Voltage VDS [V]")
plt.ylabel("Current ID [mA]")
plt.legend()  
plt.grid(True) 
plt.show()
print(f"Para VGS max = {VGS_max}:")
print("alpha =", alpha)
#print("gama =", m-2)
#print(poly_fit(VGS_max)) 

#erro
erro=[]
erro2=[]
erro.append(((new4-dfID_sat2[: largura])*100)/dfID_sat2[: largura])
#erro.append(((new6-dfID_sat2[: largura])*100)/dfID_sat2[: largura])
erro2.append(((new5-ID_max)*100)/ID_max)
#erro2.append(((new7-ID_max)*100)/ID_max)

plt.figure(13)
plt.plot(dfVG_sat2[: largura] , erro[0], label="modelo do TFT",linestyle='None',marker='.')
#plt.plot(dfVG_sat2[: largura] , erro[1], label="2º modelo ",linestyle='None',marker='.')
plt.title("")
plt.xlabel("Voltage VGS (V)")
plt.ylabel("Erro [%]")
plt.legend()  
plt.grid(True)  
plt.show()

plt.figure(14)
plt.plot(VDS_max , erro2[0], label="modelo TFT",linestyle='None',marker='.')
#plt.plot(VDS_max , erro2[1], label="2º modelo ",linestyle='None',marker='.')
plt.title("")
plt.xlabel("Voltage VDS (V)")
plt.ylabel("Erro [%]")
plt.legend()  
plt.grid(True)  
plt.show()
