from vpython import *
import numpy as np
import matplotlib.pyplot as plt
# Hard-sphere gas with moving piston and isothermal thermostat.

win = 500

Natoms = 500  # change this to have more or fewer atoms

# Typical values
L = 1 # container is a cube L on a side
gray = color.gray(0.7) # color of edges of container
mass = 4E-3/6E23 # helium mass
Ratom = 0.03 # wildly exaggerated size of helium atom
k = 1.4E-23 # Boltzmann constant
T = 300 # desired temperature (isothermal)
dt = 1E-5
ventana_pressio = []
mida_ventana = 100


animation = canvas( width=win, height=win, align='left')
animation.range = L
animation.title = 'A "hard-sphere" gas with piston'
s = """  Theoretical and averaged speed distributions (meters/sec).
  Initially all atoms have the same speed, but collisions
  change the speeds of the colliding atoms. One of the atoms is
  marked and leaves a trail so you can follow its path.
  (This simulation includes a moving piston and an isothermal thermostat.)
"""
animation.caption = s

d = L/2 + Ratom
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

# Create piston as a moving wall on the +x side
piston_thickness = 0.01
piston = box(pos=vector(d + piston_thickness/2, 0, 0),
             size=vector(piston_thickness, 2*d, 2*d), color=color.red)
piston_v = -100  # piston speed (negative = moving inward along -x)
x_min = 0
x_max = 0.75

# NEW piston dynamics variables
Mpiston = 5E-6
p_piston = Mpiston * piston_v
xpiston = piston.pos.x
vpiston = piston_v

Atoms = []
p = []
apos = []
pavg = np.sqrt(2 * mass * 1.5 * k * T)

P = []
P_teo = 0
P_tot = 0
t_acum = []
t_tot = 0
P_teorica = []
V_inv=[]
    
for i in range(Natoms):
    x = L * random() - L/2
    y = L * random() - L/2
    z = L * random() - L/2
    if i == 0:
        Atoms.append(sphere(pos=vector(x, y, z), radius=Ratom, color=color.cyan, 
                            make_trail=True, retain=100, trail_radius=0.3*Ratom))
    else:
        Atoms.append(sphere(pos=vector(x, y, z), radius=Ratom, color=gray))
    apos.append(vec(x, y, z))
    theta = pi * random()
    phi = 2 * pi * random()
    px = pavg * np.sin(theta) * np.cos(phi)
    py = pavg * np.sin(theta) * np.sin(phi)
    pz = pavg * np.cos(theta)
    p.append(vector(px, py, pz))

deltav = 100
def barx(v): return int(v/deltav)

nhisto = int(4500/deltav)
histo = [0.0] * nhisto
histo[barx(pavg/mass)] = Natoms

gg = graph(width=win, height=0.4*win, xmax=3000, align='left',
    xtitle='speed, m/s', ytitle='Number of atoms', ymax=Natoms*deltav/1000)

theory = gcurve(color=color.blue, width=2)
dv = 10
for v in range(0, 3001+dv, dv):
    theory.plot(v, (deltav/dv) * Natoms * 4 * pi * ((mass/(2 * pi * k * T))**1.5) 
                * np.exp(-0.5 * mass * (v**2) / (k * T)) * (v**2) * dv)

accum = [[deltav * (i + 0.5), 0] for i in range(int(3000/deltav))]
vdist = gvbars(color=color.red, delta=deltav)

def interchange(v1, v2):
    barx1 = barx(v1)
    barx2 = barx(v2)
    if barx1 == barx2:  return
    if barx1 >= len(histo) or barx2 >= len(histo): return
    histo[barx1] -= 1
    histo[barx2] += 1

def checkCollisions():
    hitlist = []
    r2 = (2 * Ratom)**2
    for i in range(Natoms):
        ai = apos[i]
        for j in range(i):
            aj = apos[j]
            dr = ai - aj
            if mag2(dr) < r2: 
                hitlist.append([i,j])
    return hitlist

nhisto = 0
n_steps = 1500 #ero total de pasos de simulaciónación

for step in range(n_steps):
    rate(300)
    
    impuls_pas = 0

    # Ajuste isotérmico: reescalamos las velocidades para mantener la temperatura constante
    E = sum(p[i].mag2/(2 * mass) for i in range(Natoms))
    T_current = (2/3) * (E/(Natoms * k))
    if T_current > 0:
        factor = np.sqrt(T / T_current)
        for i in range(Natoms):
            p[i] *= factor
        histo = [0.0] * len(histo)
        for i in range(Natoms):
            bin = barx(p[i].mag / mass)
            if 0 <= bin < len(histo):
                histo[bin] += 1

    # Actualización de las posiciones de las partículas
    for i in range(Natoms):
        Atoms[i].pos = apos[i] = apos[i] + (p[i]/mass) * dt

    # Comprobamos las colisiones entre partículas
    hitlist = checkCollisions()

    for ij in hitlist:
        i, j = ij
        ptot = p[i] + p[j]
        posi, posj = apos[i], apos[j]
        vi, vj = p[i]/mass, p[j]/mass
        vrel = vj - vi
        if vrel.mag2 == 0: continue
        rrel = posi - posj
        if rrel.mag > Ratom: continue
        dx = dot(rrel, norm(vrel))
        dy = cross(rrel, norm(vrel)).mag
        alpha = asin(dy/(2 * Ratom))
        dmove = (2 * Ratom) * cos(alpha) - dx
        deltat = dmove / vrel.mag
        posi -= vi * deltat
        posj -= vj * deltat
        mtot = 2 * mass
        pcmi = p[i] - ptot * mass / mtot
        pcmj = p[j] - ptot * mass / mtot
        rrel = norm(rrel)
        pcmi -= 2 * pcmi.dot(rrel) * rrel
        pcmj -= 2 * pcmj.dot(rrel) * rrel
        p[i] = pcmi + ptot * mass / mtot
        p[j] = pcmj + ptot * mass / mtot
        apos[i] = posi + (p[i]/mass) * deltat
        apos[j] = posj + (p[j]/mass) * deltat
        interchange(vi.mag, p[i].mag / mass)
        interchange(vj.mag, p[j].mag / mass)

    # Manejo de las colisiones con las paredes
    for i in range(Natoms):
        loc = apos[i]
        if loc.x < -L / 2:
            apos[i].x = -L / 2
            p[i].x = abs(p[i].x)

        # Colisión con el pistón móvil
        if loc.x + Ratom > xpiston:
            v_atom = p[i].x / mass
            v_p_old = vpiston

            v_atom_new = (v_atom * (mass - Mpiston) + 2 * Mpiston * v_p_old) / (mass + Mpiston)
            v_p_new = (2 * mass * v_atom + (Mpiston - mass) * v_p_old) / (mass + Mpiston)
            
            delta_p = mass * (v_atom_new - v_atom)
            impuls_pas += -delta_p  # Negatiu perquè l'impuls aplicat al pistó és en direcció contrària

            p[i].x = mass * v_atom_new
            p_piston = Mpiston * v_p_new
            vpiston = v_p_new

            apos[i].x = xpiston - piston_thickness / 2 - Ratom

        if abs(loc.y) > L / 2:
            apos[i].y = -L / 2 if loc.y < 0 else L / 2
            p[i].y = abs(p[i].y) if loc.y < 0 else -abs(p[i].y)
        if abs(loc.z) > L / 2:
            apos[i].z = -L / 2 if loc.z < 0 else L / 2
            p[i].z = abs(p[i].z) if loc.z < 0 else -abs(p[i].z)

    # Movimiento del pistón
    xpiston += vpiston * dt
    if xpiston < x_min:
        xpiston = x_min
        vpiston = abs(vpiston)
    elif xpiston > x_max:
        xpiston = x_max
        vpiston = -abs(vpiston)
    piston.pos.x = xpiston

    # Cálculo de la energía total y temperatura promedio
    E = sum(p[i].mag2 / (2 * mass) for i in range(Natoms))
    K_inst = 0 
    for i in range(Natoms):
        K_inst += p[i].mag2 / (2 * mass)
    
    T_inst = (2 * K_inst) / (3 * Natoms * k)

    # Cálculo de la presión en función de la energía y volumen
    A_piston = (2 * d) ** 2
    V_actual = (xpiston + L / 2) * L ** 2
    P_inst = impuls_pas / (A_piston * dt)
    inc_p = 0  # Impulso total recibido por el pistón en este paso

    # Presión total
    ventana_pressio.append(P_inst)
    if len(ventana_pressio) > mida_ventana:
        ventana_pressio.pop(0)

    P_tot = np.mean(ventana_pressio)

    # **Cálculo de la presión teórica**
    P_teo = (Natoms * k * T) / V_actual  # Usamos la ecuación de los gases ideales

    # Guardamos los datos para graficar
    if nhisto % 10 == 0:
        P.append(P_tot)
        P_teorica.append(P_teo)
        t_acum.append(t_tot)
        V_inv.append(1 / V_actual)

    # Incrementamos el tiempo total
    t_tot += dt
    nhisto += 1

# Graficamos la presión medi
plt.figure(figsize=(10, 6))
plt.plot(t_acum[2:], P[2:], label="Pressió mitjana (simulada)", color="red")  # sense marker
plt.plot(t_acum[2:], P_teorica[2:], label="Pressió teòrica", color="blue", linestyle="--")  # sense marker
plt.xlabel("Temps (s)")
plt.ylabel("Pressió (Pa)")
plt.title("Evolució de la pressió en procés isotèrmic")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gràfica de Pressió vs 1/Volum (només la simulada amb punts)
plt.figure(figsize=(10, 6))
plt.scatter(V_inv[2:], P[2:], label="Pressió simulada vs 1/V", color="green")  # amb punts
plt.plot(V_inv[2:], P_teorica[2:], label="Pressió teòrica vs 1/V", color="blue", linestyle="-")  # línia sense punts
plt.xlabel("1 / Volum (1/m³)")
plt.ylabel("Pressió (Pa)")
plt.title("Llei de Boyle: Pressió vs 1/Volum")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
