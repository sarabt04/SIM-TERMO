from vpython import *
import matplotlib.pyplot as plt
import numpy as np

#Web VPython 3.2

# Hard-sphere gas.

# Bruce Sherwood
win = 500
Natoms = 500                #Fixat a 500 atoms

#0.TIPICAL VALUES
L = 1                       # container is a cube L on a side
gray = color.gray(0.7)      # color of edges of container
mass = 4E-3/6E23            # helium mass
Ratom = 0.03                # wildly exaggerated size of helium atom
k = 1.4E-23                 # Boltzmann constant
T = 300                     # around room temperature
dt = 1E-5



#1.CREACIO DE LA CAIXA 3D (TITOL, COLOR...)---------------------------------------------------------------------------------------
animation = canvas( width=win, height=win, align='left')
animation.range = L
animation.title = 'A "hard-sphere" gas'
s = """  Theoretical and averaged speed distributions (meters/sec).
  Initially all atoms have the same speed, but collisions
  change the speeds of the colliding atoms. One of the atoms is
  marked and leaves a trail so you can follow its path.
  
"""
animation.caption = s

d = L/2+Ratom
r = 0.005
boxbottom = curve(color=gray, radius=r)
boxbottom.append([vector(-d,-d,-d), vector(-d,-d,d), vector(d,-d,d), vector(d,-d,-d), vector(-d,-d,-d)])
boxtop = curve(color=gray, radius=r)
boxtop.append([vector(-d,d,-d), vector(-d,d,d), vector(d,d,d), vector(d,d,-d), vector(-d,d,-d)])
vert1 = curve(color=gray, radius=r)
vert2 = curve(color=gray, radius=r)
vert3 = curve(color=gray, radius=r)
vert4 = curve(color=gray, radius=r)
vert1.append([vector(-d,-d,-d), vector(-d,d,-d)])
vert2.append([vector(-d,-d,d), vector(-d,d,d)])
vert3.append([vector(d,-d,d), vector(d,d,d)])
vert4.append([vector(d,-d,-d), vector(d,d,-d)])



#2.CREACIO DELS ATOMS (AMB x I p RANDOMS)-----------------------------------------------------------------------------------------

Atoms = []                      #GUARDAR EL NUM D'ATOMS PEL DIBUIX
p = []                          #LLISTA DE MOMENTS DE CADA PARTICULA
apos = []                       #LLISTA DE POSICIONS DE CADA PARTICULA
pavg = sqrt(2*mass*1.5*k*T)     # average kinetic energy p**2/(2mass) = (3/2)kT
    
for i in range(Natoms):
    x = L*random()-L/2          #cada atom te una posicio random
    y = L*random()-L/2
    z = L*random()-L/2
    if i == 0:                  #El 1r atom cyan
        Atoms.append(sphere(pos=vector(x,y,z), radius=Ratom, color=color.cyan, make_trail=True, retain=100, trail_radius=0.3*Ratom)) 
    else: Atoms.append(sphere(pos=vector(x,y,z), radius=Ratom, color=gray)) #LA RESTA D'ATOMS GRISOS
    
    apos.append(vec(x,y,z))     #guardem les posicions
    theta = pi*random()         #Angles random per tenir mom random
    phi = 2*pi*random()
    px = pavg*sin(theta)*cos(phi) 
    py = pavg*sin(theta)*sin(phi)
    pz = pavg*cos(theta)
    p.append(vector(px,py,pz))  #Guardem els moms



#3, CREACIO DE L'HISTOGRAMA-------------------------------------------------------------------------------------------------------    

deltav = 100                                         #Amplada de cada barra de l'histograma

def barx(v):
    return int(v/deltav)                             #Index de cada barra de l'histograma (de 0 a 100-1; de 100 a 200-2;...)

nhisto = int(4500/deltav)                            # numero de barres de l'histograma
histo = []
for i in range(nhisto): histo.append(0.0)            #totes les bins (barres) inicialment estan a 0 menys una (segunet linia)
histo[barx(pavg/mass)] = Natoms                      #barx(pavg/mass)=bin de v=pavg/mass=velocitat q tenen totes les particules a l'inici (es la mateixa per totes)

gg = graph( width=win, height=0.4*win, xmax=3000, align='left',                 #Creacio grafic (la x(=v) va fins a 3000)
    xtitle='speed, m/s', ytitle='Number of atoms', ymax=100)     #x=VELOCITATS; y=N ATOMS PER CADA v


#3.1PLOT DE LA DISTRIBUCIO DE MAXWELL BOLTZMANN (teoric)-----------------------------------------------------

theory = gcurve( color=color.blue, width=2 )
dv = 10
#T=300
for v in range(0,3001+dv,dv):  # theoretical prediction
    theory.plot( v, (deltav/dv)*Natoms*4*pi*((mass/(2*pi*k*T))**1.5) *exp(-0.5*mass*(v**2)/(k*T))*(v**2)*dv )

#T=100
theory = gcurve( color=color.green, width=2 )
T=100
for v in range(0,3001+dv,dv):  # theoretical prediction
    theory.plot( v, (deltav/dv)*Natoms*4*pi*((mass/(2*pi*k*T))**1.5) *exp(-0.5*mass*(v**2)/(k*T))*(v**2)*dv )
    
T=300

#3.2PLOT DE LAES VELOCITATS REALS ACTUALITZADES AMB LA SIMULACIO---------------------------------------------

accum = []                                                                     #Mitjana del num d'atoms a cada bin acumulada amb t             
for i in range(int(3000/deltav)): accum.append([deltav*(i+.5),0])              #De cada bin [centre, 0 q es fara servir per fer una mitjana]
vdist = gvbars(color=color.red, delta=deltav )



#4.ACTUALITZACIO DE L'HISTOGRAMA--------------------------------------------------------------------------------------------------

#4.1FUNCIONS PER ACTUALITZAR L'HISTOGRAMA--------------------------------------------------------------------
def interchange(v1, v2):  # ELIMINEM UN ATOM DEL BIN V1 I EL POSEM AL BIN V2 (quan 2 atoms canvien de v (en una colisio p.ex.))
    barx1 = barx(v1)
    barx2 = barx(v2)
    if barx1 == barx2:  return
    if barx1 >= len(histo) or barx2 >= len(histo): return
    histo[barx1] -= 1
    histo[barx2] += 1

    
def checkCollisions():  # DETECCIO DEL NUM DE COLISIONS
    hitlist = []
    r2 = 2*Ratom
    r2 *= r2                                                        #r2=r_atom^2 (minima dist pq 2 atoms es toquin) (fem servir els quadrats pq hem de fer menys operacions)
    for i in range(Natoms):                                         #per cada atom i agafem la seva posicio ai
        ai = apos[i]
        for j in range(i) :                                         #mirem la resta de particules a quina distancia estan [mag2(dr)=dist al quadrat entre 2 posicions]
            aj = apos[j]
            dr = ai - aj
            if mag2(dr) < r2: hitlist.append([i,j])                 #si estan mes a prop q la dist minima pq colisionin, contem la colisio
    return hitlist


#4.2DINAMICA MOLECULAR---------------------------------------------------------------------------------------
nhisto = 0 # number of histogram snapshots to average (Snapshot="captura" de l'histograma de velocitats per cada t)
t_tot=0
P_tot=0
P_teo = 0
t_acum=[]
P=[]
P_teorica=[]
T=[]
info = wtext(text="Valors pressió\n")


while t_tot<0.01: #BUCLE Q ES REPETEIX INFINITAMENT PER ANAR ACTUALITZANT LA POSICIO I LA VELOCITAT DELS ATOMS
    rate(300)  
    t_tot+=dt                                                                #Fa q el bucle s'actualitzi max 300cops cada seg                
    
    # Accumulate and average histogram snapshots
    for i in range(len(accum)):                                                #Actualitzem el 2n valor de la llista acum
        accum[i][1] = (nhisto*accum[i][1] + histo[i])/(nhisto+1)               #Mitjana progressiva de l'histograma: (valor anterior+actual, dividit pel nombre de snapshots
        
    if nhisto % 10 == 0:
        vdist.data = accum                                                     #Cada 10iteracions actualitzem l'histograma
    nhisto += 1                                                                #Sumem una snapshot

    # ACTUALITZACIO DE POSICIONS (movem cada atom segons la seva v=pavg/m)
    for i in range(Natoms): 
        Atoms[i].pos = apos[i] = apos[i] + (p[i]/mass)*dt
        
    ##############################   APARTAT 2:PROCES ISOCOR, NOVES v RANDOM A LES PARTICULES   ################################
    
    
    #ASSSIGNEM UNA v RANDOM Q SEGUEIXI LA DISTRIBUCIÓ DE MB
    #Definim la funció que seguiran
    def box_muller(mu=0,sig=1):
        u1=np.random.uniform(0,1)
        u2=np.random.uniform(0,1)
        z1=np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)        #variables gaussianes
        z2=np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
        return sig*z1, sig*z2                               #retorn de 2 variables gaussianes independents
    
    def v_random_MB(T):
        sigma=np.sqrt(k*T/mass)                             #sigma de MB
        v_i, _=box_muller(0,sigma)
        return v_i                                          #la funcio ens retorna una v random         

    #A CADA PAS DE t AGAFEM  UNA PARTICULA RANDOM I LI CANVIEM A SEVA v ALEATORIAMENT 
    for _ in range(10):
        i_random = np.random.randint(0, Natoms)                 #particula random que agafem
        v_inicial=p[i_random]/mass
        
        #Canvi de v
        T_termostat=100
        v_new_x = v_random_MB(T_termostat)
        v_new_y = v_random_MB(T_termostat)
        v_new_z = v_random_MB(T_termostat)
        v_final=np.sqrt(v_new_x**2+v_new_y**2+v_new_z**2)
        p_final=mass*v_final
        
        #Canviem el moment de la particula (pero mantenint la direcció)
        p[i_random]=p[i_random]*p_final/mag(p[i_random])
        interchange(v_inicial.mag , p[i_random].mag/mass) 
    
    # DETECTEM QUINS ATOMS ESTAN COLISIONANT
    hitlist = checkCollisions()

    # PER AQEULLES PAPRTICULES QUE HAN XOOCAT (elasticament) ACTUALITZEM ELS MOMENTS DELS 2 ATOMS
    for ij in hitlist:                                                          #Actualitza les posicions fent lo de particules (SL-SCM-SL)
        i = ij[0]
        j = ij[1]
        ptot = p[i]+p[j]
        posi = apos[i]
        posj = apos[j]
        vi = p[i]/mass
        vj = p[j]/mass
        vrel = vj-vi
        a = vrel.mag2
        if a == 0: continue;  # exactly same velocities
        rrel = posi-posj
        if rrel.mag > Ratom: continue # one atom went all the way through another
    
        # theta is the angle between vrel and rrel:
        dx = dot(rrel, vrel.hat)       # rrel.mag*cos(theta)
        dy = cross(rrel, vrel.hat).mag # rrel.mag*sin(theta)
        # alpha is the angle of the triangle composed of rrel, path of atom j, and a line
        #   from the center of atom i to the center of atom j where atome j hits atom i:
        alpha = asin(dy/(2*Ratom)) 
        d = (2*Ratom)*cos(alpha)-dx # distance traveled into the atom from first contact
        deltat = d/vrel.mag         # time spent moving from first contact to position inside atom
        
        posi = posi-vi*deltat # back up to contact configuration
        posj = posj-vj*deltat
        mtot = 2*mass
        pcmi = p[i]-ptot*mass/mtot # transform momenta to cm frame
        pcmj = p[j]-ptot*mass/mtot
        rrel = norm(rrel)
        pcmi = pcmi-2*pcmi.dot(rrel)*rrel # bounce in cm frame
        pcmj = pcmj-2*pcmj.dot(rrel)*rrel
        p[i] = pcmi+ptot*mass/mtot # transform momenta back to lab frame
        p[j] = pcmj+ptot*mass/mtot
        apos[i] = posi+(p[i]/mass)*deltat # move forward deltat in time
        apos[j] = posj+(p[j]/mass)*deltat
        interchange(vi.mag, p[i].mag/mass)                                     #Actualitzem l'histograma si les v han canviat de bin
        interchange(vj.mag, p[j].mag/mass)
           
    #ACTUALITZEM LA POSICIÓ I LA VELOCITAT DE LES PARTICULES QUE HAN XOCAT CONTRA LA PARET (xoc perfectamet elastic)
    inc_p=0
    p_i_xocs=[]
    for i in range(Natoms):
        loc = apos[i]
        
    ##############################   APARTAT 2:PROCES ISOCOR, CÀLCUL DE L'EVOLUCIO DE P i T   ##################################        
        if abs(loc.x) > L/2:
            p_i_xocs.append(abs(p[i].x))
            if loc.x < 0: p[i].x =  abs(p[i].x)
            else: p[i].x =  -abs(p[i].x)
                
        if abs(loc.y) > L/2:
            p_i_xocs.append(abs(p[i].y))
            if loc.y < 0: p[i].y = abs(p[i].y)
            else: p[i].y =  -abs(p[i].y)
        
        if abs(loc.z) > L/2:
            p_i_xocs.append(abs(p[i].z))
            if loc.z < 0: p[i].z =  abs(p[i].z)
            else: p[i].z =  -abs(p[i].z)
        
    for j in p_i_xocs:
        inc_p=inc_p+2*j
            
    P_inst=inc_p/(dt*6*L**2)
    P_tot=(nhisto*P_tot+P_inst)/(nhisto+1)
    if nhisto % 10 == 0:
        P.append(P_tot) 
        t_acum.append(t_tot)
        
            
    ##############################   APARTAT 2:PROCES ISOCOR, CALCUL DE LA PRESSIO TEORICA   ##################################
    K_inst = 0 
    for i in range(Natoms):
        K_inst += mag(p[i])**2/(2*mass)

    T_inst = (2*K_inst)/(3*Natoms*k)
    P_inst = Natoms*k*T_inst                        #No dividim entre L^3 pq és 1
    P_teo = (nhisto*P_teo + P_inst)/(nhisto+1)
    
    if nhisto % 10 == 0:
       P_teorica.append(P_teo)
       T.append(T_inst)
       info.text = f"Temps total: {t_tot:.4f} s\nPressió simulada: {P_tot:.2e} s\nPressió teòrica: {P_teo:.2e} s"



plt.plot(t_acum,P, color='c', label='sim.(dsim)')
plt.plot(t_acum, P_teorica, color='blue', label='teo.(dsim)')
plt.xlabel(r't [s]')
plt.ylabel(r'P [Pa]')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.show()     

plt.plot(t_acum,T, color='c', label='sim.(dsim)')
plt.xlabel(r't [s]')
plt.ylabel(r'T [K]')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.show()     