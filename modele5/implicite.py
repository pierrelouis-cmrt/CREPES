# script_backEuler.py
# -------------------------------------------------
# 0-D climat : méthode implicite Backward Euler
# -------------------------------------------------
import numpy as np, matplotlib.pyplot as plt
S0, σ, α = 1361, 5.67e-8, 0.25
lat, Tatm, C, dt = np.radians(49), 253.15, 1e8, 1800
def δ(n): return np.radians(23.44)*np.sin(2*np.pi*(284+n)/365)
def cosi(n,h): δn=δ(n); H=np.radians(15*(h-12)); return max(np.sin(lat)*np.sin(δn)+np.cos(lat)*np.cos(δn)*np.cos(H),0)
def φ(n,h): return S0*cosi(n,h)*(1-α)
def loop(days):
    N=int(days*86400/dt); t=np.arange(N+1)*dt; T=np.empty(N+1); T[0]=288
    for k in range(N):
        n,h=int(t[k]//86400)+1,(t[k]%86400)/3600
        Φ=φ(n,h)
        X=T[k]
        for _ in range(8):                      # Newton
            f=(Φ+σ*Tatm**4-σ*X**4)/C
            F=X-T[k]-dt*f
            dF=1-dt*(-4*σ*X**3/C)
            X-=F/dF
            if abs(F)<1e-6: break
        T[k+1]=X
    return t, T
for d,ttl in [(1,'1 jour'),(365,'1 an')]:
    tt,TT=loop(d); plt.figure(); plt.plot(tt/86400,TT-273.15); plt.title(f'Backward Euler – {ttl}'); plt.xlabel('Jour'); plt.ylabel('°C'); plt.grid(); plt.tight_layout(); plt.show()
